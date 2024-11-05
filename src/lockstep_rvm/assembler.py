from lockstep_rvm.vm import Instr
from lockstep_rvm.vm import Deque


def parse_pred_args(args: list[str]) -> list[str]:
    pred_args = []
    if args[0] == "^":
        pred_args.append("^")
        args = args[1:]
    for i, arg in enumerate(args):
        if i % 2 == 0:
            pred_args.append(arg[2])
        else:
            pred_args.append(arg[0])
    return pred_args


def assemble(rasm_file_name: str) -> list[Instr]:
    """
    Create a list of `Instr` objects from a file containing regex assembly code.
    """
    with open(rasm_file_name) as f:
        prog = []
        for line in f:
            if line == "\n":
                continue
            line = line.replace(",", "")
            op, *args = line.split()
            if op == "char":
                # remove the quotes surrounding the character
                args = [args[0][1]]
            elif op == "pred":
                args = parse_pred_args(args)
            prog.append(Instr(op, args))
    return prog


def preprocess_labels(prog):
    label_to_index = {}
    instrs = []
    for instr in prog:
        if instr.op.startswith(".t"):
            label_to_index[instr.op.strip(":")] = len(instrs)
        else:
            instrs.append(instr)
            print(instr)

    def convert_labels_to_indices(args):
        return [
            str(label_to_index[arg]) if arg in label_to_index.keys() else str(arg)
            for arg in args
        ]

    for i, instr in enumerate(instrs):
        match instr.op:
            case "jmp" | "split" | "tswitch":
                instr.args = convert_labels_to_indices(instr.args)
            case _:
                pass
    print(instrs)
    return instrs
