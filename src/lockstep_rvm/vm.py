import sys

from typing import TypeVar

ITEM = TypeVar("ITEM")
K = TypeVar("K")
V = TypeVar("V")


class Instr:
    """An instruction for the regex virtual machine (VM)."""

    def __init__(self, op: str, args: list[str]):
        self.op = op
        self.args = args

    def __repr__(self):
        def arg_to_str(arg):
            return f"'{arg}'" if self.op == "char" else arg

        args_str = ", ".join(map(arg_to_str, self.args))
        return f"{self.op} {args_str}".strip()


class SharedDict(dict[K, V]):
    def __init__(self, *args, ref_count=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_count = ref_count

    def new_copy(self):
        print(f"Copying... (ref_count == {self.ref_count})")
        return SharedDict(super().copy(), ref_count=1)


class Thread:
    """A metaphorical thread for the VM to execute."""

    def __init__(
        self,
        ip,
        pending_save: SharedDict[int, int],
        curr_saves: list[int],
        prev_saves: list[int],
    ):
        self.ip = ip
        self.save = pending_save
        self.curr_saves = curr_saves
        self.prev_saves = prev_saves

    def __repr__(self):
        return f"Thread(ip={self.ip}, pending_save={self.save})"


class Deque(list[ITEM]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append(self, item):
        super().append(item)

    def prepend(self, item):
        self.insert(0, item)

    def pop_back(self):
        return self.pop()

    def pop_front(self):
        return self.pop(0)

    def __repr__(self):
        return f"Deque({super().__repr__()})"

import copy
class State:
    def __init__(self, threads: Deque[Thread]):
        self.ipq = tuple(thread.ip for thread in threads)
        self.threads = tuple(copy.deepcopy(threads))

    def __repr__(self):
        return f"State(ips={self.ipq})"

    def __hash__(self):
        return hash(self.ipq)

    def __eq__(self, other):
        for i in range(len(self.ipq)):
            if self.ipq[i] != other.ipq[i]:
                return False
        return len(self.ipq) == len(other.ipq)

    def is_dead(self):
        return len(self.threads) == 0


def commit_saves(pending_save: SharedDict[int, int], save: list[int]):
    """Write saves to the shared buffer"""
    print(f"writing saves ({len(pending_save)})")
    for i, sp in pending_save.items():
        save[i] = sp


def write(pending_save: SharedDict[int, int], j: int, sp: int):
    if pending_save.ref_count > 1:
        pending_save.ref_count -= 1
        pending_save = pending_save.new_copy()
    pending_save[j] = sp
    return pending_save


def run(prog: list[Instr], text: str) -> list[int]:
    """Run `prog` with the `text` input and return the matched text (or `None`)."""
    text = text + "\0"
    q, next_q = Deque[Thread](), Deque[Thread]()
    max_save_id = max(int(instr.args[0]) for instr in prog if instr.op == "save") + 1
    num_epssets = sum(1 for instr in prog if instr.op == "epsset")
    eps_sp = []
    thread: Thread = Thread(0, SharedDict[int, int](), [], [])
    next_q.append(thread)
    sp = -1
    # For tracing the VM's execution
    line_executed = ""
    comment = ""

    next_eps_sp = [-1] * num_epssets

    match_thread = dict[State, Thread]()
    ns_cache = dict[State, dict[str, State]]()

    state, next_state = None, State(next_q)
    prev_match_thread = None

    while not next_state.is_dead():
        print(f"{line_executed:<20} {comment}")
        line_executed = ""
        comment = ""

        prev_state = state
        state = next_state
        next_state = None

        sp += 1
        print(f"\n; sp advanced (`sp` = {sp})\n")

        if state in ns_cache.keys() and text[sp] in ns_cache[state].keys():
            print(f"Cache hit: {state} {text[sp]}")
            next_state = ns_cache[state][text[sp]]

            # write captures
            for thread in next_state.threads:
                for j in thread.prev_saves:
                    print(f"setting pending_save[{j}] = {sp}")
                    thread.save = write(thread.save, j, sp)
            mt = match_thread.get(state)
            if mt is not None:
                for j in mt.prev_saves:
                    mt.save = write(mt.save, j, sp)
                prev_match_thread = mt
            continue

        q, next_q = next_q, q
        next_q.clear()

        eps_sp, next_eps_sp = next_eps_sp, eps_sp
        next_eps_sp = [-1] * num_epssets

        marked = [False] * len(prog)

        while len(q) > 0:
            print(f"{line_executed:<20} {comment}")
            thread = q.pop_front()
            if marked[thread.ip]:
                line_executed = f"{thread.ip:>2}  {prog[thread.ip]}"
                print(
                    f"{line_executed:<20} ; thread killed (ip {thread.ip} already visited)"
                )
                thread.save.ref_count -= 1
                line_executed = ""
                comment = ""
                continue
            marked[thread.ip] = True
            instr = prog[thread.ip]
            line_executed = f"{thread.ip:>2}  {instr}"
            comment = ""
            match instr.op:
                case "jmp":
                    thread.ip = int(instr.args[0])
                    q.prepend(thread)
                    continue
                case "split":
                    d1, d2 = map(int, instr.args)
                    q.prepend(
                        Thread(
                            d2,
                            thread.save.new_copy(),
                            thread.curr_saves.copy(),
                            thread.prev_saves.copy(),
                        )
                    )
                    q.prepend(
                        Thread(
                            d1,
                            thread.save.new_copy(),
                            thread.curr_saves.copy(),
                            thread.prev_saves.copy(),
                        )
                    )
                    thread.save.ref_count += 1
                    del thread
                    continue
                case "tswitch":
                    j = int(instr.args[0])
                    for i in reversed(range(j)):
                        t = Thread(
                            int(instr.args[i + 1]),
                            thread.save.new_copy(),
                            thread.curr_saves.copy(),
                            thread.prev_saves.copy(),
                        )
                        thread.save.ref_count += 1
                        q.prepend(t)
                    thread.save.ref_count -= 1
                    del thread
                    continue

                case "char":
                    if text[sp] != instr.args[0]:
                        comment = (
                            f"; thread killed (r'{text[sp]}' != r'{instr.args[0]}')\n"
                        )
                        thread.save.ref_count -= 1
                        del thread
                    else:
                        thread.ip += 1
                        thread.prev_saves = thread.curr_saves.copy()
                        thread.curr_saves = []
                        thread.save = thread.save.new_copy()
                        next_q.append(thread)
                        next_eps_sp = eps_sp.copy()
                        comment = f"; added ip `{thread.ip}` to `next_q` (`*sp == {instr.args[0]}`)'\n"
                    continue

                case "save":
                    j = int(instr.args[0])
                    thread.save = write(thread.save, j, sp)
                    thread.curr_saves.append(j)
                case "match":
                    q.clear()
                    print(f"{line_executed:<20} {comment}")
                    thread.save = thread.save.new_copy()
                    thread.curr_saves = thread.curr_saves.copy()
                    thread.prev_saves = thread.prev_saves.copy()
                    prev_match_thread = thread
                    break

                case "end":
                    if text[sp] != "\0":
                        comment = f"; thread killed (sp != len(text))\n"
                        thread.save.ref_count -= 1
                        del thread
                        continue
                case "begin":
                    if sp != 0:
                        comment = f"; thread killed (sp != 0)\n"
                        thread.save.ref_count -= 1
                        del thread
                        continue
                case "epsset":
                    j = int(instr.args[0]) // 8
                    eps_sp[j] = sp
                    pass
                case "epschk":
                    j = int(instr.args[0]) // 8
                    if eps_sp[j] == sp:
                        comment = f"; thread killed (eps_sp[{j}] == sp == {sp})\n"
                        thread.save.ref_count -= 1
                        continue
                    pass
                case _:
                    raise ValueError(f"Unknown instruction: {instr.op}")
            thread.ip += 1
            q.prepend(thread)
        if prev_match_thread is not None:
            match_thread[state] = prev_match_thread
        if not state in ns_cache.keys():
            ns_cache[state] = dict[str, State]()
        if prev_state in ns_cache.keys():
            assert prev_state is not None
            ns_cache[prev_state][text[sp - 1]] = state
        next_state = State(next_q)
    match_save = [-1] * max_save_id
    if prev_match_thread is not None:
        commit_saves(prev_match_thread.save, match_save)
    return match_save


def prog_to_str(prog: list[Instr]) -> str:
    """Return a string representation of the program."""
    return "\n".join(str(instr) for i, instr in enumerate(prog))


def print_captures(query_text: str, saves: list[int]):
    for i in range(len(saves) // 2):
        start, stop = saves[2 * i], saves[2 * i + 1]

        if start == -1 and stop == -1:
            print(f"Group {i} not captured")
        else:
            print(f"Group {i} [{start}:{stop}]: {query_text[start:stop]}")


if __name__ == "__main__":
    import sys
    from lockstep_rvm import assembler

    try:
        rasm_file_name, text = sys.argv[1], sys.argv[2]
    except IndexError:
        print("Usage: python -m lockstep_rvm.vm <rasm_file_name> <text_to_match>")
        sys.exit(1)
    prog = assembler.assemble(rasm_file_name)

    assert prog_to_str(prog) == open(rasm_file_name).read()

    saves = run(prog, text)

    print()
    print_captures(text, saves)
    print()
