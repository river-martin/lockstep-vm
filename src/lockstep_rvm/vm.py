import sys
import logging
from typing import TypeVar

ITEM = TypeVar("ITEM")
K = TypeVar("K")
V = TypeVar("V")

GREY_START = "\033[90m"
COLOUR_RESET = "\033[0m"

logging.basicConfig(format="%(message)s")
logger = logging.getLogger(__name__)


def log(
    *values: object,
    sep=" ",
    end="",
) -> None:
    msg = sep.join(map(str, values)) + end
    logger.debug(msg)


class Instr:
    """An instruction for the regex virtual machine (VM)."""

    def __init__(self, op: str, args: list[str]):
        self.op = op
        self.args = args

    def __repr__(self):
        def arg_to_str(arg):
            return f"'{arg}'" if self.op == "char" else arg

        if self.op == "pred":
            if self.args[0] == "^":
                args_str = (
                    "^["
                    + ", ".join(
                        f"({self.args[i]}, {self.args[i+1]})"
                        for i in range(1, len(self.args), 2)
                    )
                    + "]"
                )
            else:
                args_str = (
                    "["
                    + ", ".join(
                        f"({self.args[i]}, {self.args[i+1]})"
                        for i in range(0, len(self.args), 2)
                    )
                    + "]"
                )
        else:
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
    """A DFA state, with an ordered list of threads to execute."""

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


def exec_prev_saves(state: State, sp: int, perf_metrics):
    """For each thread in the state, execute the saves for the previous step"""
    for i, thread in enumerate(state.threads):
        log(f"executing prev saves for thread {i}")
        for j in thread.prev_save_id:
            log(f"setting save[{j}] = {sp}")
            thread.save[j] = sp
        perf_metrics["n_save_writes"] += len(thread.prev_save_id)


def exec_saves(state: State, sp: int, perf_metrics):
    """For each thread in the state, execute the saves for the current step"""
    for i, thread in enumerate(state.threads):
        log(f"executing saves for thread {i}")
        for j in thread.save_id:
            log(f"setting save[{j}] = {sp}")
            thread.save[j] = sp
        perf_metrics["n_save_writes"] += len(thread.save_id)


def update_match(thread: Thread, sp: int, perf_metrics) -> Thread:
    """Execute the saves for the match thread"""
    for sid in thread.save_id:
        thread.save[sid] = sp
    perf_metrics["n_save_writes"] += len(thread.save_id)
    return thread


def run(prog: list[Instr], text: str, use_cache):
    """Run `prog` with the `text` input and return the matched text (or `None`)."""
    q, next_q = Deque[Thread](), Deque[Thread]()
    max_save_id = (
        max(
            int(instr.args[0])
            for instr in prog
            if instr.op == "save" and type(instr.args[0]) == str
        )
        + 1
    )
    thread: Thread = Thread(0, dict[int, int](), [], [])
    next_q.append(thread)
    sp = -1
    # For tracing the VM's execution
    line_executed = ""
    comment = ""

    state_to_match_thread = dict[State, Thread]()
    ns_cache = dict[State, dict[str, State]]()

    state, next_state = None, State(next_q)
    prev_match_thread = None
    skipped_saves_and_match = False

    perf_metrics = {"n_cache_hits": 0, "n_save_buf_copy": 0, "n_save_writes": 0}

    while not next_state.is_dead():
        log(f"{line_executed:<20} {comment}")
        line_executed = ""
        comment = ""

        prev_state = state
        state = next_state
        next_state = None

        sp += 1
        log(f"\n; sp advanced (`sp` = {sp})\n")

        can_use_cache = (
            not sp == len(text) - 1 and use_cache
        )  # we can't use the cache if we're at the end of the text
        if (
            can_use_cache
            and state in ns_cache.keys()
            and sp < len(text)
            and text[sp] in ns_cache[state].keys()
        ):
            next_state = ns_cache[state][text[sp]]
            if next_state == state and text[sp] == text[sp + 1]:
                # Captures and match would be overwritten in the next step, so we can skip them
                skipped_saves_and_match = True
            else:
                # Captures and match might not be overwritten in the next step, so we need to execute them
                exec_prev_saves(next_state, sp, perf_metrics)
                mthread = state_to_match_thread.get(state)
                if mthread is not None:
                    prev_match_thread = update_match(mthread, sp, perf_metrics)
                skipped_saves_and_match = False
            log(f"Cache hit: {state} {text[sp]}")
            perf_metrics["n_cache_hits"] += 1
            continue

        if skipped_saves_and_match:
            # We skipped the saves and match in the previous step, so we need to execute them now
            exec_prev_saves(state, sp - 1, perf_metrics)
            mthread = state_to_match_thread.get(state)
            if mthread is not None:
                prev_match_thread = update_match(mthread, sp - 1, perf_metrics)
            skipped_saves_and_match = False

        q, next_q = Deque(state.threads), q
        next_q.clear()
        for i, thread in enumerate(q):
            assert thread.ip == state.threads[i].ip

        marked = [False] * len(prog)

        while len(q) > 0:
            kill_thread = False
            reason = None

            log(f"{line_executed:<20} {comment}")
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
                        perf_metrics["n_save_buf_copy"] += 2
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
                        perf_metrics["n_save_buf_copy"] += j
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

                    case "pred":
                        neg = instr.args[0] == "^"
                        interval_bounds = instr.args[1:] if neg else instr.args
                        for i in range(0, len(interval_bounds), 2):
                            start, stop = interval_bounds[i], interval_bounds[i + 1]
                            if ord(start) <= ord(text[sp]) <= ord(stop):
                                if neg:
                                    kill_thread = True
                                    reason = f"pred mismatch @ sp={sp}"
                                    break
                                else:
                                    thread.ip += 1
                                    thread.prev_save_id = thread.save_id
                                    thread.save_id = []
                                    thread.save = thread.save
                                    next_q.append(thread)
                                    comment = f"; added ip `{thread.ip}` to `next_q` (`{start} <= *sp <= {stop}`)\n"
                                    break

                    case "save":
                        j = int(instr.args[0])
                        thread.save[j] = sp
                        thread.save_id.append(j)
                        perf_metrics["n_save_writes"] += 1

                    case "match":
                        q.clear()
                        log(f"{line_executed:<20} {comment}")
                        prev_match_thread = thread
                        prev_match_thread.prev_save_id = prev_match_thread.save_id
                        break

                    case "end":
                        if sp != len(text):
                            kill_thread = True
                            reason = f"sp != len(text)"
                    case "begin":
                        if sp != 0:
                            kill_thread = True
                            reason = f"sp != 0"
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
            log(
                f"Saving match thread: {prev_match_thread}, {prev_match_thread.save_id}"
            )
            state_to_match_thread[state] = prev_match_thread
        if not state in ns_cache.keys():
            ns_cache[state] = dict[str, State]()
        if prev_state in ns_cache.keys():
            assert prev_state is not None
            ns_cache[prev_state][text[sp - 1]] = state
        next_state = State(next_q)
        if (
            state in ns_cache.keys()
            and sp < len(text)
            and text[sp] in ns_cache[state].keys()
        ):
            ns = ns_cache[state][text[sp]]
            # Check that the cache hit is consistent with the next state (i.e. prev_save_id is the same)
            # If a different path (with different prev_save_id) can lead to the same state, then the cache is invalid
            assert len(ns.threads) == len(next_state.threads)
            for i, thread in enumerate(ns.threads):
                assert thread.prev_save_id == next_state.threads[i].prev_save_id
    match_save = [-1] * max_save_id
    if prev_match_thread is not None:
        for sid, sp in prev_match_thread.save.items():
            match_save[sid] = sp
    return match_save, perf_metrics


def prog_to_str(prog: list[Instr]) -> str:
    """Return a string representation of the program."""
    return "\n".join(str(instr) for i, instr in enumerate(prog))


def print_captures(query_text: str, saves: list[int]):
    max_span_width = 0
    for i in range(len(saves) // 2):
        start, stop = saves[2 * i], saves[2 * i + 1]
        max_span_width = max(max_span_width, len(f"{start}-{stop}"))
    max_group_width = len(f"Group {len(str(len(saves) // 2 - 1))}")
    for i in range(len(saves) // 2):
        start, stop = saves[2 * i], saves[2 * i + 1]

        if start != -1 and stop != -1:
            group_ident = f"Group {i}"
            group_span = f"{start}-{stop}"
            group_text = (
                query_text[start:stop]
                if start != stop
                else GREY_START + "empty string" + COLOUR_RESET
            )
            print(
                f"{group_ident:<{max_group_width}}  {GREY_START}{group_span:<{max_span_width}}{COLOUR_RESET}  | {group_text}"
            )


def print_stats(perf_metrics, text):
    print(f"Cache hits: {perf_metrics['n_cache_hits']}")
    print(f"Save buffer copies: {perf_metrics['n_save_buf_copy']}")
    print(f"Save buffer writes: {perf_metrics['n_save_writes']}")
    print(f"Text length: {len(text)}")


if __name__ == "__main__":
    import sys
    from lockstep_rvm import assembler

    try:
        rasm_file_name, text, log_level, use_cache = (
            sys.argv[1],
            sys.argv[2],
            sys.argv[3],
            sys.argv[4] == "cache",
        )
    except IndexError:
        print(
            "Usage: python -m lockstep_rvm.vm <rasm_file_name> <text_to_match> <debug|info> <cache|no_cache>"
        )
        sys.exit(1)
    match log_level:
        case "debug":
            logger.setLevel(logging.DEBUG)
        case "info":
            logger.setLevel(logging.INFO)
        case _:
            raise ValueError("log_level must be either 'debug' or 'info'")

    prog = assembler.assemble(rasm_file_name)

    if prog_to_str(prog) != open(rasm_file_name).read():
        print(prog_to_str(prog), open(rasm_file_name).read(), sep="\n")
    assert prog_to_str(prog) == open(rasm_file_name).read()

    saves, perf_metrics = run(prog, text, use_cache)

    print_captures(text, saves)
    print_stats(perf_metrics, text)
