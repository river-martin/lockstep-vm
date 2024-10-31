from lockstep_rvm.vm import Instr
from lockstep_rvm.vm import Deque


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
            prog.append(Instr(op, args))
    return prog


def reverse(prog: list[Instr]) -> list[Instr]:
    """
    Reverse the program.
    """
    prog = [Instr("prog_start", [])] + prog
    targets = {}
    rev_target_count = 0
    prog_with_labels = []
    for i, instr in enumerate(prog):
        match instr.op:
            case "split":

                prog_with_labels.append(Instr(f".t{rev_target_count}:", []))

                if instr.args[0] not in targets.keys():
                    targets[instr.args[0]] = [rev_target_count]
                else:
                    targets[instr.args[0]].append(rev_target_count)

                if instr.args[1] not in targets.keys():
                    targets[instr.args[1]] = [rev_target_count]
                else:
                    targets[instr.args[1]].append(rev_target_count)

                rev_target_count += 1

                prog_with_labels.append(Instr("nop", []))


            case "jmp":

                prog_with_labels.append(Instr(f".t{rev_target_count}:", []))

                if instr.args[0] not in targets.keys():
                    targets[instr.args[0]] = [rev_target_count]
                else:
                    targets[instr.args[0]].append(rev_target_count)

                rev_target_count += 1

                prog_with_labels.append(Instr("nop", []))

            case "match":
                prog_with_labels.append(Instr("nop", []))

            case "end":
                prog_with_labels.append(Instr("begin", []))

            case "prog_start":
                prog_with_labels.append(Instr("match", []))

            case "epschk":
                prog_with_labels.append(Instr("epsset", instr.args))

            case "epsset":
                prog_with_labels.append(Instr("epschk", instr.args))

            case _:
                prog_with_labels.append(instr)

    rev = Deque()
    i = 0
    for instr in prog_with_labels:
        if instr.op.startswith(".t"):
            rev.prepend(instr)
        else:
            if str(i) in targets.keys():
                rev.prepend(
                    Instr(
                        f"tswitch",
                        [str(len(targets[str(i)]))]
                        + [f".t{j}" for j in targets[str(i)]],
                    )
                )
            elif instr.op == "prog_start":
                rev.prepend(Instr("match", []))
            elif instr.op != "nop":
                rev.prepend(instr)
            i += 1
    return rev


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
