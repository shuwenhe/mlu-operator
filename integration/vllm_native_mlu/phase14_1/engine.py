"""Frozen prompt driver for the actual Vendor engine."""
import argparse
import faulthandler
from array import array
import json
import os
import time
from pathlib import Path

import torch_mlu
from vllm import LLM, SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError

from .plugin import ARCH
from .boundary_audit import measure
from .faults import FAULTS


def close_model(worker):
    model = worker.model_runner.model
    model.close()
    return {"rank": model.rank, "device": str(model.device), "released": True}


def set_fault(worker, kind):
    return worker.model_runner.model.arm_acceptance_fault(kind)


def run_negative(llm, frozen, kind):
    ranks = llm.collective_rpc(set_fault,args=(kind,))
    assert sorted(ranks)==[0,1]
    ids,_ = prompt(frozen,1)
    try:
        llm.generate([{"prompt_token_ids":ids}],SamplingParams(temperature=0,max_tokens=1),use_tqdm=False)
    except EngineDeadError as exc:
        print(json.dumps({"phase14_1":"negative_engine_dead","fault":kind,
                          "message":str(exc)}),flush=True)
        return False
    except RuntimeError as exc:
        label = ("Rank schedule digest mismatch" if kind=="divergent" else
                 "Peer metadata preflight failed" if kind=="missing-metadata" else
                 "Scheduler validation failed across ranks")
        if label not in str(exc):
            raise
        pending = list(llm.llm_engine.output_processor.external_req_ids)
        if pending:
            llm.llm_engine.abort_request(pending)
        print(json.dumps({"phase14_1":"negative_rejected","fault":kind,
                          "aborted_requests":len(pending),"message":str(exc)}),flush=True)
    else:
        raise AssertionError(f"Scheduler accepted invalid fixture: {kind}")
    return True


def prompt(root,length):
    index = (1,7,17,65).index(length)
    case = root/f"request{index}"
    ids = array("I")
    ids.frombytes((case/"prompt.ids").read_bytes())
    if len(ids)!=length:
        raise ValueError("Frozen prompt length mismatch")
    return list(ids),case


def expected_token(path):
    logits=array("f")
    logits.frombytes(path.read_bytes())
    return max(range(len(logits)),key=logits.__getitem__)


def run_group(llm, frozen, lengths, count, offset):
    prompts,cases = zip(*(prompt(frozen,length) for length in lengths))
    started=time.perf_counter()
    outputs=llm.generate([{"prompt_token_ids":ids} for ids in prompts],
                         SamplingParams(temperature=0,max_tokens=count,ignore_eos=True),use_tqdm=False)
    elapsed=(time.perf_counter()-started)*1000
    assert len(outputs)==len(lengths)
    for size,ids,case,result in zip(lengths,prompts,cases,outputs):
        targets=[case/f"{step}-logits.bin" for step in range(count)]
        tokens=[expected_token(p) for p in targets]
        got=list(result.outputs[0].token_ids)
        print(json.dumps({"phase14_1":"generate","prompt_length":size,"prompt_ids":ids,
                          "tokens":got,"expected":tokens,"batch_size":len(lengths)}),flush=True)
        if got!=tokens:
            raise ValueError(f"Frozen greedy parity failed at length {size}: {got} != {tokens}")
    dump=Path(os.environ["NEURX_TP2_DUMP"])
    stride=102400*4
    for step in range(count):
        references=[(case/f"{step}-logits.bin").read_bytes() for case in cases]
        replicated=[]
        for rank in (0,1):
            raw=(dump/f"rank{rank}_logits{offset+step}.bin").read_bytes()
            if len(raw)!=len(lengths)*stride:
                raise ValueError("Engine logit batch order/shape does not match grouped fixture")
            replicated.append(raw)
            unused=set(range(len(references)))
            for row in range(len(lengths)):
                actual=raw[row*stride:(row+1)*stride]
                comparisons=[(i,measure(actual,references[i],"fp32")) for i in sorted(unused)]
                matches=[(i,m) for i,m in comparisons if not m["tolerance_mismatch"]]
                if not matches:
                    result=min(comparisons,key=lambda pair:pair[1]["tolerance_mismatch"])
                    print(json.dumps({"phase14_1":"logits_failed","rank":rank,"step":step,**result[1]}),flush=True)
                    raise ValueError("Frozen logits tolerance exceeded")
                i,matched=matches[0]
                unused.remove(i)
                print(json.dumps({"phase14_1":"logits_parity","rank":rank,"device":f"mlu:{rank}",
                                  "step":step,"prompt_length":lengths[i],**matched}),flush=True)
        if replicated[0]!=replicated[1]:
            raise ValueError("Replicated worker logits differ")
    print(json.dumps({"phase14_1":"group_wall","lengths":lengths,"tokens":count,"elapsed_ms":elapsed}),flush=True)
    return offset+count


def cross_page(llm,frozen):
    ids,case=prompt(frozen,1)
    result, = llm.generate([{"prompt_token_ids":ids}],
                           SamplingParams(temperature=0,max_tokens=17,ignore_eos=True),use_tqdm=False)
    tokens=list(result.outputs[0].token_ids)
    expected=[expected_token(case/f"{step}-logits.bin") for step in range(4)]
    if len(tokens)!=17 or tokens[:4]!=expected:
        raise ValueError("Cross-page lifecycle prompt failed the frozen token prefix")
    print(json.dumps({"phase14_1":"cross_page_completed","tokens":len(tokens),
                      "frozen_prefix_tokens":4}),flush=True)

    choices=(7,65)
    prompts,cases=zip(*(prompt(frozen,length) for length in choices))
    results=llm.generate([{"prompt_token_ids":ids} for ids in prompts],
                         SamplingParams(temperature=0,max_tokens=4,ignore_eos=True),use_tqdm=False)
    for length,case,result in zip(choices,cases,results):
        tokens=list(result.outputs[0].token_ids)
        expected=[expected_token(case/f"{step}-logits.bin") for step in range(4)]
        if tokens!=expected:
            raise ValueError(f"Chunked mixed lifecycle token mismatch at length {length}")
        print(json.dumps({"phase14_1":"mixed_lifecycle_tokens","prompt_length":length,
                          "tokens":tokens,"expected":expected}),flush=True)


def inject_worker_loss(worker):
    import torch

    model = worker.model_runner.model
    print(json.dumps({"phase14_1":"worker_loss_entered","rank":model.rank}),flush=True)
    if model.rank==1:
        os._exit(71)
    value=torch.tensor([1.0],device=model.device)
    return model.group.all_reduce(value).item()


def exercise_worker_loss(llm):
    start=time.perf_counter()
    try:
        llm.collective_rpc(inject_worker_loss,timeout=15)
    except (TimeoutError,RuntimeError) as exc:
        elapsed=time.perf_counter()-start
        if elapsed>25:
            raise ValueError("Collective failure exceeded the test deadline") from exc
        print(json.dumps({"phase14_1":"worker_loss_detected","elapsed_s":elapsed,
                          "error_type":type(exc).__name__,"message":str(exc)}),flush=True)
    else:
        raise AssertionError("Collective unexpectedly completed without its peer")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length",type=int,choices=(1,7,17,65),default=1)
    parser.add_argument("--tokens",type=int,choices=(2,4),default=2)
    run_mode=parser.add_mutually_exclusive_group()
    run_mode.add_argument("--expanded",action="store_true")
    run_mode.add_argument("--negative",choices=FAULTS)
    run_mode.add_argument("--negative-suite",action="store_true")
    run_mode.add_argument("--cross-page",action="store_true")
    run_mode.add_argument("--worker-loss",action="store_true")
    args = parser.parse_args()
    frozen = Path("/project/artifacts/build/phase12_5_4/native-dump-yarn-scale")
    llm = LLM(model="/checkpoint",trust_remote_code=True,dtype="bfloat16",tensor_parallel_size=2,
              hf_overrides={"architectures":[ARCH]},enforce_eager=True,max_model_len=128,
              block_size=16,max_num_seqs=2,max_num_batched_tokens=16 if args.cross_page else 128,
              num_gpu_blocks_override=32,
              enable_prefix_caching=False,gpu_memory_utilization=0.45,disable_log_stats=True,
              disable_custom_all_reduce=True)
    from ..phase12_3.vendor_trace import enable_trace
    enable_trace()
    model_available = True
    try:
        if args.worker_loss:
            exercise_worker_loss(llm)
        elif args.negative:
            model_available = run_negative(llm,frozen,args.negative)
        elif args.negative_suite:
            for kind in FAULTS:
                model_available = run_negative(llm,frozen,kind)
                if not model_available:
                    break
        elif args.cross_page:
            cross_page(llm,frozen)
        elif args.expanded:
            offset=0
            for length in (7,17,65,1,7,17,65,1,7,17):
                offset=run_group(llm,frozen,(length,),4,offset)
            run_group(llm,frozen,(7,17),4,offset)
        else:
            run_group(llm,frozen,(args.length,),args.tokens,0)
    finally:
        try:
            if not args.worker_loss and model_available:
                print(json.dumps({"phase14_1":"model_release","ranks":llm.collective_rpc(close_model)}),flush=True)
        finally:
            if args.negative or args.negative_suite:
                faulthandler.dump_traceback_later(45)
            try:
                llm.llm_engine.engine_core.shutdown()
            finally:
                if args.negative or args.negative_suite:
                    faulthandler.cancel_dump_traceback_later()
            print(json.dumps({"phase14_1":"engine_shutdown"}),flush=True)


if __name__=="__main__":
    main()
