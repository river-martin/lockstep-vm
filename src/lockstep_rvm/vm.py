import sys
import copy

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


class Thread:
    """A metaphorical thread for the VM to execute."""

    def __init__(
        self,
        ip,
        save: dict[int, int],
        save_id: list[int],
        prev_save_id: list[int],
    ):
        self.ip = ip
        self.save = save
        self.save_id = save_id
        self.prev_save_id = prev_save_id

    def __repr__(self):
        return f"Thread(ip={self.ip}, save={self.save})"


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


class State:
    def __init__(self, threads: Deque[Thread]):
        self.ipq = tuple(thread.ip for thread in threads)
        self.threads = tuple(threads)

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


def exec_prev_saves(thread, sp):
    for i in thread.prev_save_id:
        print(f"setting save[{i}] = {sp}")
        thread.save[i] = sp


def exec_saves(thread, sp):
    for i in thread.save_id:
        print(f"setting save[{i}] = {sp}")
        thread.save[i] = sp


def run(prog: list[Instr], text: str) -> list[int]:
    """Run `prog` with the `text` input and return the matched text (or `None`)."""
    q, next_q = Deque[Thread](), Deque[Thread]()
    max_save_id = max(int(instr.args[0]) for instr in prog if instr.op == "save") + 1
    thread: Thread = Thread(0, dict[int, int](), [], [])
    next_q.append(thread)
    sp = -1
    # For tracing the VM's execution
    line_executed = ""
    comment = ""

    match_thread = dict[State, Thread]()
    ns_cache = dict[State, dict[str, State]]()

    state, next_state = None, State(next_q)
    prev_match_thread = None
    used_cache = False

    while not next_state.is_dead():
        print(f"{line_executed:<20} {comment}")
        line_executed = ""
        comment = ""

        prev_state = state
        state = next_state
        next_state = None

        sp += 1
        print(f"\n; sp advanced (`sp` = {sp})\n")

        # What if a cache hit occurs but the path is different? i.e. what if the prev_save_id is different?
        can_use_cache = not sp == len(text) - 1 # we can't use the cache if we're at the end of the text
        if (
            can_use_cache and 
            state in ns_cache.keys()
            and sp < len(text)
            and text[sp] in ns_cache[state].keys()
        ):
            next_state = ns_cache[state][text[sp]]
            print(f"Cache hit: {state} {text[sp]}")
            used_cache = True
            continue


            # if (state == next_state and sp + 1 < len(text) and text[sp + 1] == text[sp]):
            #     print(f"Cache hit: {state} {text[sp]}")
            #     # # write captures
            #     # for thread in next_state.threads:
            #     #     print(thread.prev_save_id)
            #     #     exec_prev_saves(thread, sp)
            #     used_cache = True
            #     continue
            # if not (state == next_state and sp + 1 < len(text) and text[sp + 1] == text[sp]):
            #     print(state == next_state, sp + 1 < len(text))
            #     # write captures
            #     # for thread in next_state.threads:
            #     #     print(thread.prev_save_id)
            #     #     exec_prev_saves(thread, sp)
            #     if sp + 1 < len(text):
            #         print(f"Cache hit: {state} {text[sp]}")
            #         used_cache = True
            #         continue

            #     # for thread in state.threads:
            #     #     exec_saves(thread, sp-2)
            #     #     exec_prev_saves(thread, sp-2)

            #     # mt = match_thread.get(state)
            #     # if mt is not None:
            #     #     exec_saves(mt, sp)
            #     #     prev_match_thread = mt
                
            #     pass
        if used_cache:
            for thread in state.threads:
                exec_prev_saves(thread, sp-1)
            used_cache = False

        q, next_q = Deque(state.threads), q
        next_q.clear()
        for i, thread in enumerate(q):
            assert thread.ip == state.threads[i].ip

        marked = [False] * len(prog)

        while len(q) > 0:
            kill_thread = False
            reason = None

            print(f"{line_executed:<20} {comment}")
            thread = q.pop_front()
            if marked[thread.ip]:
                kill_thread = True
                reason = f"ip {thread.ip} already visited"
            else:
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
                                thread.save.copy(),
                                thread.save_id.copy(),
                                thread.prev_save_id.copy(),
                            )
                        )
                        q.prepend(
                            Thread(
                                d1,
                                thread.save.copy(),
                                thread.save_id.copy(),
                                thread.prev_save_id.copy(),
                            )
                        )
                        del thread
                        continue

                    case "tswitch":
                        j = int(instr.args[0])
                        for i in reversed(range(j)):
                            t = Thread(
                                int(instr.args[i + 1]),
                                thread.save.copy(),
                                thread.save_id.copy(),
                                thread.prev_save_id.copy(),
                            )
                            q.prepend(t)
                        del thread
                        continue

                    case "char":
                        if sp == len(text) or text[sp] != instr.args[0]:
                            kill_thread = True
                            reason = f"char mismatch @ sp={sp}"
                        else:
                            thread.ip += 1
                            thread.prev_save_id = thread.save_id
                            thread.save_id = []
                            thread.save = thread.save
                            next_q.append(thread)
                            comment = f"; added ip `{thread.ip}` to `next_q` (`*sp == {instr.args[0]}`)'\n"
                            continue

                    case "save":
                        j = int(instr.args[0])
                        thread.save[j] = sp
                        thread.save_id.append(j)

                    case "match":
                        q.clear()
                        print(f"{line_executed:<20} {comment}")
                        prev_match_thread = thread
                        prev_match_thread.prev_save_id = prev_match_thread.save_id
                        break

                    case "end":
                        if sp != len(text):
                            comment = f"; thread killed (sp != len(text))\n"
                            del thread
                            continue
                    case "begin":
                        if sp != 0:
                            comment = f"; thread killed (sp != 0)\n"
                            del thread
                            continue
                    case "epsset":
                        pass
                    case "epschk":
                        pass
                    case _:
                        raise ValueError(f"Unknown instruction: {instr.op}")
            if kill_thread:
                comment = f"; thread killed ({reason})\n"
                del thread
                continue
            thread.ip += 1
            q.prepend(thread)
        if prev_match_thread is not None:
            print(
                f"Saving match thread: {prev_match_thread}, {prev_match_thread.save_id}"
            )
            match_thread[state] = prev_match_thread
        if not state in ns_cache.keys():
            ns_cache[state] = dict[str, State]()
        if prev_state in ns_cache.keys():
            assert prev_state is not None
            ns_cache[prev_state][text[sp - 1]] = state            
        next_state = State(next_q)
        if state in ns_cache.keys() and sp < len(text) and text[sp] in ns_cache[state].keys():
            ns = ns_cache[state][text[sp]]
            assert len(ns.threads) == len(next_state.threads)
            for i, thread in enumerate(ns.threads):
                assert thread.prev_save_id == next_state.threads[i].prev_save_id
    match_save = [-1] * max_save_id
    if prev_match_thread is not None:
        for sid, sp in prev_match_thread.save.items():
            match_save[sid] = sp
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
