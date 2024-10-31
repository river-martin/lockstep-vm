import numpy as np
import sys
import copy


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


class SharedDict(dict):
    def __init__(self, *args, ref_count=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_count = ref_count

    def new_copy(self):
        print(f"Copying... (ref_count == {self.ref_count})")
        return SharedDict(super().copy(), ref_count=1)


class Thread:
    """A metaphorical thread for the VM to execute."""

    def __init__(
        self, ip, pending_save: SharedDict, saves_in_current_step, saves_in_last_step
    ):
        self.ip = ip
        self.pending_save: SharedDict = pending_save
        self.lsaves = saves_in_last_step
        self.csaves = saves_in_current_step

    def __repr__(self):
        return f"Thread(ip={self.ip}, pending_save={self.pending_save})"

    def copy(self):
        self.pending_save.ref_count += 1
        return Thread(
            self.ip,
            self.pending_save,
            self.csaves,
            self.lsaves,
        )


class Deque(list):
    def __init__(self):
        super().__init__()

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

    def copy(self):
        c = Deque()
        for item in self:
            c.append(item.copy())
        return c


class State:
    def __init__(self, threads: Deque):
        self.ips = tuple(thread.ip for thread in threads)
        self.threads = threads.copy()

    def __repr__(self):
        return f"State(ips={self.ips})"

    def __hash__(self):
        return hash(self.ips)

    def __eq__(self, other):
        for i in range(len(self.ips)):
            if self.ips[i] != other.ips[i]:
                return False
        return len(self.ips) == len(other.ips)

    def is_dead(self):
        return len(self.threads) == 0


def commit_saves(pending_save: SharedDict, save: list[int]):
    """Write saves to the shared buffer"""
    print(f"writing saves ({len(pending_save)})")
    for i, sp in pending_save.items():
        save[i] = sp
    pending_save.clear()


def run(prog: list[Instr], text: str) -> tuple[str | None, list[int]]:
    """Run `prog` with the `text` input and return the matched text (or `None`)."""
    query_text = text + "\0"
    q = Deque()
    next_q = Deque()
    num_saves = max(int(instr.args[0]) for instr in prog if instr.op == "save") + 1
    num_epssets = sum(1 for instr in prog if instr.op == "epsset")
    eps_sp = []
    thread: Thread = Thread(0, SharedDict(), [], [])
    next_q.append(thread)
    sp = -1
    prev_match = None
    prev_saves: list[int] = []
    # For tracing the VM's execution
    line_executed = ""
    comment = ""
    save: list[int] = [-1] * num_saves

    next_eps_sp = [-1] * num_epssets

    cache = {}

    next_state = State(next_q)

    while not next_state.is_dead():
        print(f"{line_executed:<20} {comment}")
        line_executed = ""
        comment = ""
        
        state, next_state = next_state, None
        next_q, q = q, state.threads
        parallel_branch_count = len(q)
        sp += 1
        print(f"\n; sp advanced (`sp` = {sp})\n")
        next_q.clear()

        eps_sp, next_eps_sp = next_eps_sp, eps_sp
        next_eps_sp = [-1] * num_epssets

        marked = [False] * len(prog)

        if hash(state) in cache.keys() and query_text[sp] in cache[hash(state)].keys():
            next_state = cache[hash(state)][query_text[sp]]
            print(hash(state), hash(next_state))
            print(query_text[sp], query_text[sp+1])
            # write captures
            for thread in next_state.threads:
                for j in thread.lsaves:
                    print(f"setting pending_save[{j}] = {sp}")
                    thread.pending_save[j] = sp
                        
            print(f"Cache hit: {state} {query_text[sp]}")
            continue
        next_thread_id = 0
        while len(q) > 0:

            print(f"{line_executed:<20} {comment}")
            thread = q.pop_front()
            if marked[thread.ip]:
                line_executed = f"{thread.ip:>2}  {prog[thread.ip]}"
                print(
                    f"{line_executed:<20} ; thread killed (ip {thread.ip} already visited)"
                )
                thread.pending_save.ref_count -= 1
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
                            d2, thread.pending_save, thread.csaves.copy(), thread.lsaves.copy()
                        )
                    )
                    q.prepend(
                        Thread(
                            d1, thread.pending_save, thread.csaves.copy(), thread.lsaves.copy()
                        )
                    )
                    thread.pending_save.ref_count += 1
                    continue
                case "char":
                    if sp >= len(text) or text[sp] != instr.args[0]:
                        comment = f"; thread killed (r'{r'\0' if sp >= len(text) else text[sp]}' != r'{instr.args[0]}')\n"
                        thread.pending_save.ref_count -= 1
                    else:
                        thread.ip += 1
                        if (
                            len(q) == 0
                            and parallel_branch_count == 1
                            and len(thread.pending_save) > 0
                        ):
                            commit_saves(thread.pending_save, save)
                        thread.lsaves, thread.csaves = thread.csaves, []
                        next_q.append(thread)
                        next_eps_sp = eps_sp.copy()
                        comment = f"; added ip `{thread.ip}` to `next_q` (`*sp == {instr.args[0]}`)'\n"
                    next_thread_id += 1
                    continue
                case "epsset":
                    j = int(instr.args[0]) // 8
                    eps_sp[j] = sp

                case "epschk":
                    j = int(instr.args[0]) // 8
                    if eps_sp[j] == sp:
                        comment = f"; thread killed (eps_sp[{j}] == sp == {sp})\n"
                        thread.pending_save.ref_count -= 1
                        continue
                    pass

                case "save":
                    j = int(instr.args[0])
                    if parallel_branch_count == 1:
                        save[j] = sp
                    else:
                        if thread.pending_save.ref_count > 1:
                            thread.pending_save.ref_count -= 1
                            thread.pending_save = thread.pending_save.new_copy()
                        thread.pending_save[j] = sp
                        thread.csaves.append(j)
                case "match":
                    match_save = copy.deepcopy(save)
                    print(thread.pending_save)
                    for j in thread.lsaves:
                        thread.pending_save[j] = sp - 1
                    if len(next_q) == 0:
                        commit_saves(thread.pending_save, match_save)
                    prev_saves = copy.deepcopy(save)
                    q.clear()
                    print(f"{line_executed:<20} {comment}")
                    break
                case "tswitch":
                    j = int(instr.args[0])
                    for i in reversed(range(j)):
                        thread = Thread(int(instr.args[i + 1]), thread.pending_save, thread.csaves.copy(), thread.lsaves.copy())
                        thread.pending_save.ref_count += 1
                        q.prepend(thread)
                    continue
                case "end":
                    if sp != len(text):
                        comment = f"; thread killed (sp != len(text))\n"
                        thread.pending_save.ref_count -= 1
                        del thread
                        continue
                case "begin":
                    if sp != 0:
                        comment = f"; thread killed (sp != 0)\n"
                        thread.pending_save.ref_count -= 1
                        del thread
                        continue
                case _:
                    raise ValueError(f"Unknown instruction: {instr.op}")
            thread.ip += 1
            q.prepend(thread)
        if hash(state) not in cache.keys():
            cache[hash(state)] = {}
        next_state = State(next_q)
        cache[hash(state)][query_text[sp]] = next_state
    return prev_match, prev_saves


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
        rasm_file_name, query_text = sys.argv[1], sys.argv[2]
    except IndexError:
        print("Usage: python -m lockstep_rvm.vm <rasm_file_name> <text_to_match>")
        sys.exit(1)
    prog = assembler.assemble(rasm_file_name)

    assert prog_to_str(prog) == open(rasm_file_name).read()

    _, saves = run(prog, query_text)

    print()
    print_captures(query_text, saves)
    print()
