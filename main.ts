// Sokoban solver for make.fun's "Build the best Sokoban solver, 21 November 2026".
//
// Strategy: the levels are a hallway with independent rooms off it (one door
// each, 3-4 boxes per room), so the level is split into its rooms and each room
// is solved on its own with a push-based A* search. A push moves one box one
// square; the player walks between pushes. The search state is the player's
// square plus the sorted box squares. Its estimate is the cheapest assignment
// of boxes to goals over walking distances that ignore boxes (admissible, so
// A* returns the fewest pushes). States are pruned when a box sits on a square
// no pull can ever bring back to a goal (a reverse-pull deadlock table), when
// four boxes lock into a 2x2 block, or when some box cannot reach any goal.
// The rooms' move strings are stitched together with hallway walks, and the
// whole answer is replayed on the full level before it is written.
//
// Determinism is by work, never by the clock: every budget below is a fixed
// count of state expansions, every iteration order is fixed, and nothing here
// reads the time, randomness, or the environment for variation. A room the
// search cannot solve yields an empty answer for that instance rather than
// an illegal or partial one.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed work budgets (state expansions), the same on every run.
const ASTAR_BUDGET = 250000;
const GREEDY_BUDGET = 150000;

type Dir = { dx: number; dy: number; ch: string };
const DIRS: Dir[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

function inside(w: number, h: number, x: number, y: number): boolean {
  return x >= 0 && y >= 0 && x < w && y < h;
}

/** Walking distances from one square over the squares marked in `walkable`. */
function bfsDistances(w: number, h: number, walkable: Uint8Array, from: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  dist[from] = 0;
  const queue: number[] = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inside(w, h, nx, ny)) continue;
      const n = ny * w + nx;
      if (walkable[n] === 0 || dist[n] >= 0) continue;
      dist[n] = dist[c]! + 1;
      queue.push(n);
    }
  }
  return dist;
}

/** The player's walk from `from` to `to` over walkable squares avoiding `blocked`, or null. */
function bfsPath(
  w: number,
  h: number,
  walkable: Uint8Array,
  blocked: Uint8Array,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-1);
  const prevCh = new Array<string>(w * h).fill('');
  prev[from] = from;
  const queue: number[] = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inside(w, h, nx, ny)) continue;
      const n = ny * w + nx;
      if (walkable[n] === 0 || blocked[n] === 1 || prev[n] >= 0) continue;
      prev[n] = c;
      prevCh[n] = d.ch;
      if (n === to) {
        let out = '';
        let cur = n;
        while (cur !== from) {
          out = prevCh[cur]! + out;
          cur = prev[cur]!;
        }
        return out;
      }
      queue.push(n);
    }
  }
  return null;
}

/** Every square the player can reach from `from` over walkable squares avoiding `blocked`. */
function reachableSet(
  w: number,
  h: number,
  walkable: Uint8Array,
  blocked: Uint8Array,
  from: number,
): Uint8Array {
  const seen = new Uint8Array(w * h);
  seen[from] = 1;
  const queue: number[] = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inside(w, h, nx, ny)) continue;
      const n = ny * w + nx;
      if (walkable[n] === 0 || blocked[n] === 1 || seen[n] === 1) continue;
      seen[n] = 1;
      queue.push(n);
    }
  }
  return seen;
}

/**
 * Dead squares of a region: a box on one can never reach a goal again. Found
 * by reverse pulls — a square is live when a box there could have been pulled
 * from a goal, one pull at a time, with the player's stand square open. A
 * square this marks dead is dead under any player walk, so pruning on it is
 * safe; squares it leaves live may still be dead, which only costs search.
 * The entry square counts as open floor for the stand squares.
 */
function deadTable(
  w: number,
  h: number,
  region: Uint8Array,
  entry: number,
  goals: number[],
): Uint8Array {
  const live = new Uint8Array(w * h);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  const standOpen = (c: number): boolean => region[c] === 1 || c === entry;
  for (let qi = 0; qi < queue.length; qi++) {
    const b = queue[qi]!;
    const bx = b % w;
    const by = (b / w) | 0;
    for (const d of DIRS) {
      const ax = bx - d.dx;
      const ay = by - d.dy;
      const sx = bx - 2 * d.dx;
      const sy = by - 2 * d.dy;
      if (!inside(w, h, ax, ay) || !inside(w, h, sx, sy)) continue;
      const a = ay * w + ax;
      if (region[a] === 0 || live[a] === 1) continue;
      if (!standOpen(sy * w + sx)) continue;
      live[a] = 1;
      queue.push(a);
    }
  }
  const dead = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (region[i] === 1 && live[i] === 0) dead[i] = 1;
  return dead;
}

/** Cheapest assignment of boxes to goals over the per-goal walking distances. */
function heuristic(boxes: number[], goalDist: Int32Array[], goals: number[]): number {
  const n = boxes.length;
  if (n <= 4) {
    // Brute-force assignment (at most 4! orders), in a fixed order.
    let best = Infinity;
    const perm: number[] = goals.map((_, i) => i);
    const rec = (k: number, acc: number): void => {
      if (acc >= best) return;
      if (k === n) {
        if (acc < best) best = acc;
        return;
      }
      for (let j = k; j < n; j++) {
        const tmp = perm[k]!;
        perm[k] = perm[j]!;
        perm[j] = tmp;
        const d = goalDist[perm[k]!]![boxes[k]!]!;
        if (d >= 0) rec(k + 1, acc + d);
        const tmp2 = perm[k]!;
        perm[k] = perm[j]!;
        perm[j] = tmp2;
      }
    };
    rec(0, 0);
    return best;
  }
  let sum = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (let g = 0; g < goals.length; g++) {
      const d = goalDist[g]![b]!;
      if (d >= 0 && d < m) m = d;
    }
    if (m === Infinity) return Infinity;
    sum += m;
  }
  return sum;
}

type Push = { box: number; dir: number };
type SearchResult = { pushes: Push[]; endPlayer: number; endBoxes: number[] } | null;

type SearchCtx = {
  w: number;
  h: number;
  walkable: Uint8Array;
  region: Uint8Array;
  dead: Uint8Array;
  goalDist: Int32Array[];
  goals: number[];
  goalSet: Uint8Array;
};

/** A push-based search over one region. `mode` picks A* (fewest pushes) or greedy. */
function searchRegion(
  ctx: SearchCtx,
  startPlayer: number,
  startBoxes: number[],
  mode: 'astar' | 'greedy',
  budget: number,
): SearchResult {
  const { w, h, walkable, region, dead, goalDist, goals, goalSet } = ctx;
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b] === 1);
  const keyOf = (p: number, bs: number[]): string => p + '|' + bs.join(',');

  // The states, in discovery order: player, boxes, cost, parent, push that made it.
  const sp: number[] = [startPlayer];
  const sb: number[][] = [startBoxes.slice().sort((a, b) => a - b)];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const pushBox: number[] = [-1];
  const pushDir: number[] = [-1];
  const best = new Map<string, number>();
  best.set(keyOf(startPlayer, sb[0]!), 0);

  // Binary heap of [f, g, seq, id]: lowest f first, then deeper g, then earliest seq.
  const heap: number[][] = [];
  let seq = 0;
  const less = (a: number[], b: number[]): boolean =>
    a[0]! < b[0]! ||
    (a[0]! === b[0]! && (a[1]! > b[1]! || (a[1]! === b[1]! && a[2]! < b[2]!)));
  const heapPush = (e: number[]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): number[] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  const h0 = heuristic(sb[0]!, goalDist, goals);
  if (h0 === Infinity) return null;
  heapPush([mode === 'astar' ? h0 : h0, 0, seq++, 0]);

  let expanded = 0;
  while (heap.length > 0) {
    const entry = heapPop();
    const id = entry[3]!;
    const g = entry[1]!;
    if (best.get(keyOf(sp[id]!, sb[id]!)) !== id) continue;
    if (g !== sg[id]) continue;
    expanded++;
    if (expanded > budget) return null;
    const player = sp[id]!;
    const boxes = sb[id]!;
    if (solved(boxes)) {
      const pushes: Push[] = [];
      let cur = id;
      while (parent[cur]! >= 0) {
        pushes.push({ box: pushBox[cur]!, dir: pushDir[cur]! });
        cur = parent[cur]!;
      }
      pushes.reverse();
      return { pushes, endPlayer: player, endBoxes: boxes.slice() };
    }
    const blocked = new Uint8Array(w * h);
    for (const b of boxes) blocked[b] = 1;
    const reach = reachableSet(w, h, walkable, blocked, player);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let di = 0; di < DIRS.length; di++) {
        const d = DIRS[di]!;
        const tx = bx + d.dx;
        const ty = by + d.dy;
        if (!inside(w, h, tx, ty)) continue;
        const dest = ty * w + tx;
        if (region[dest] === 0 || blocked[dest] === 1) continue;
        const px = bx - d.dx;
        const py = by - d.dy;
        if (!inside(w, h, px, py)) continue;
        if (reach[py * w + px] === 0) continue;
        const nextBoxes = boxes.slice();
        nextBoxes[bi] = dest;
        nextBoxes.sort((a, b2) => a - b2);
        let bad = false;
        for (const nb of nextBoxes) if (dead[nb] === 1) { bad = true; break; }
        if (bad) continue;
        // A 2x2 block of boxes can never move again; dead unless all four sit on goals.
        const nblocked = new Uint8Array(w * h);
        for (const nb of nextBoxes) nblocked[nb] = 1;
        for (const nb of nextBoxes) {
          const nx = nb % w;
          const ny = (nb / w) | 0;
          if (nx + 1 < w && ny + 1 < h) {
            const q = nb + 1;
            const r = nb + w;
            const s = nb + w + 1;
            if (nblocked[q] === 1 && nblocked[r] === 1 && nblocked[s] === 1) {
              if (!(goalSet[nb] === 1 && goalSet[q] === 1 && goalSet[r] === 1 && goalSet[s] === 1)) {
                bad = true;
                break;
              }
            }
          }
        }
        if (bad) continue;
        const hh = heuristic(nextBoxes, goalDist, goals);
        if (hh === Infinity) continue;
        const g2 = g + 1;
        const key = keyOf(b, nextBoxes);
        const prevId = best.get(key);
        if (prevId !== undefined && sg[prevId]! <= g2) continue;
        const nid = sp.length;
        sp.push(b);
        sb.push(nextBoxes);
        sg.push(g2);
        parent.push(id);
        pushBox.push(b);
        pushDir.push(di);
        best.set(key, nid);
        const f = mode === 'astar' ? g2 + hh : hh;
        heapPush([f, g2, seq++, nid]);
      }
    }
  }
  return null;
}

/** Turn a push list into walk moves, replaying as it goes. Returns null if a walk is impossible. */
function pushesToMoves(
  ctx: SearchCtx,
  startPlayer: number,
  startBoxes: number[],
  pushes: Push[],
): { moves: string; endPlayer: number; endBoxes: number[] } | null {
  const { w, h, walkable } = ctx;
  let player = startPlayer;
  let boxes = startBoxes.slice().sort((a, b) => a - b);
  let moves = '';
  for (const p of pushes) {
    const blocked = new Uint8Array(w * h);
    for (const b of boxes) blocked[b] = 1;
    const d = DIRS[p.dir]!;
    const behind = p.box - d.dy * w - d.dx;
    const path = bfsPath(w, h, walkable, blocked, player, behind);
    if (path === null) return null;
    moves += path + d.ch;
    const dest = p.box + d.dy * w + d.dx;
    boxes = boxes.map((b) => (b === p.box ? dest : b)).sort((a, b2) => a - b2);
    player = p.box;
  }
  return { moves, endPlayer: player, endBoxes: boxes };
}

/** Replay moves on a region: every step legal, every box on a goal at the end. */
function verifyRegion(
  ctx: SearchCtx,
  startPlayer: number,
  startBoxes: number[],
  moves: string,
): boolean {
  const { w, h, walkable, region, goalSet } = ctx;
  const box = new Uint8Array(w * h);
  for (const b of startBoxes) box[b] = 1;
  let player = startPlayer;
  for (let k = 0; k < moves.length; k++) {
    const d = DIRS.find((dd) => dd.ch === moves[k]);
    if (d === undefined) return false;
    const next = player + d.dy * w + d.dx;
    if (walkable[next] === 0) return false;
    if (box[next] === 1) {
      const beyond = next + d.dy * w + d.dx;
      if (region[beyond] === 0 || box[beyond] === 1) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < w * h; i++) if (box[i] === 1 && goalSet[i] === 0) return false;
  return true;
}

type Room = { cells: number[]; boxes: number[]; goals: number[]; entry: number };

/** Solve one room (or, as a fallback, a whole level) from its entry square. */
function solveRegion(
  level: Level,
  cells: number[],
  boxes: number[],
  goals: number[],
  entry: number,
): { moves: string; endPlayer: number; endBoxes: number[] } | null {
  const { w, h } = level;
  const region = new Uint8Array(w * h);
  for (const c of cells) region[c] = 1;
  const walkable = new Uint8Array(w * h);
  for (const c of cells) walkable[c] = 1;
  walkable[entry] = 1;
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  const goalDist = goals.map((g) => bfsDistances(w, h, region, g));
  const dead = deadTable(w, h, region, entry, goals);
  const ctx: SearchCtx = { w, h, walkable, region, dead, goalDist, goals, goalSet };
  const sortedBoxes = boxes.slice().sort((a, b) => a - b);
  let found = searchRegion(ctx, entry, sortedBoxes, 'astar', ASTAR_BUDGET);
  if (found === null) found = searchRegion(ctx, entry, sortedBoxes, 'greedy', GREEDY_BUDGET);
  if (found === null) return null;
  const built = pushesToMoves(ctx, entry, sortedBoxes, found.pushes);
  if (built === null) return null;
  if (!verifyRegion(ctx, entry, sortedBoxes, built.moves)) return null;
  return built;
}

/**
 * Split a level into its rooms: the hallway is the player's row, and each
 * room is a connected component of the open squares off that row holding
 * boxes. Returns null when the level does not split cleanly, and the caller
 * then solves the whole level as one region.
 */
function decompose(level: Level): Room[] | null {
  const { w, h, wall } = level;
  const hy = (level.player / w) | 0;
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  let coveredBoxes = 0;
  for (let i = 0; i < w * h; i++) {
    if (wall[i] === 1 || ((i / w) | 0) === hy || seen[i] === 1) continue;
    const comp: number[] = [];
    const compSet = new Uint8Array(w * h);
    const queue: number[] = [i];
    seen[i] = 1;
    compSet[i] = 1;
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi]!;
      comp.push(c);
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (!inside(w, h, nx, ny) || ((ny === hy))) continue;
        const n = ny * w + nx;
        if (wall[n] === 1 || seen[n] === 1) continue;
        seen[n] = 1;
        compSet[n] = 1;
        queue.push(n);
      }
    }
    comp.sort((a, b) => a - b);
    const cboxes = level.boxes.filter((b) => compSet[b] === 1);
    if (cboxes.length === 0) continue;
    const cgoals = level.goals.filter((g) => compSet[g] === 1);
    if (cboxes.length !== cgoals.length) return null;
    let entry = -1;
    outer: for (const c of comp) {
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (!inside(w, h, nx, ny) || ny !== hy) continue;
        const n = ny * w + nx;
        if (wall[n] === 0) {
          entry = n;
          break outer;
        }
      }
    }
    if (entry < 0) return null;
    coveredBoxes += cboxes.length;
    rooms.push({ cells: comp, boxes: cboxes, goals: cgoals, entry });
  }
  if (rooms.length === 0 || coveredBoxes !== level.boxes.length) return null;
  rooms.sort((a, b) => a.entry - b.entry);
  return rooms;
}

/** Replay moves on the whole level: legal steps and every box on a goal. */
function verifyFull(level: Level, moves: string): boolean {
  const { w, h, wall } = level;
  const box = new Uint8Array(w * h);
  for (const b of level.boxes) box[b] = 1;
  const goalSet = new Uint8Array(w * h);
  for (const g of level.goals) goalSet[g] = 1;
  let player = level.player;
  for (let k = 0; k < moves.length; k++) {
    const d = DIRS.find((dd) => dd.ch === moves[k]);
    if (d === undefined) return false;
    const nx = (player % w) + d.dx;
    const ny = ((player / w) | 0) + d.dy;
    if (!inside(w, h, nx, ny)) return false;
    const next = ny * w + nx;
    if (wall[next] === 1) return false;
    if (box[next] === 1) {
      const bx = nx + d.dx;
      const by = ny + d.dy;
      if (!inside(w, h, bx, by)) return false;
      const beyond = by * w + bx;
      if (wall[beyond] === 1 || box[beyond] === 1) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < w * h; i++) if (box[i] === 1 && goalSet[i] === 0) return false;
  return true;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.player < 0 || level.boxes.length === 0) return '';
  if (level.boxes.length !== level.goals.length) return '';
  const { w, h, wall } = level;
  const openAll = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (wall[i] === 0) openAll[i] = 1;

  const rooms = decompose(level);
  if (rooms !== null) {
    let moves = '';
    let curPlayer = level.player;
    let curBoxes = level.boxes.slice();
    for (const room of rooms) {
      const blocked = new Uint8Array(w * h);
      for (const b of curBoxes) blocked[b] = 1;
      const walk = bfsPath(w, h, openAll, blocked, curPlayer, room.entry);
      if (walk === null) return '';
      moves += walk;
      const solved = solveRegion(level, room.cells, room.boxes, room.goals, room.entry);
      if (solved === null) return '';
      moves += solved.moves;
      curPlayer = solved.endPlayer;
      const roomBoxSet = new Set(room.boxes);
      curBoxes = curBoxes
        .filter((b) => !roomBoxSet.has(b))
        .concat(solved.endBoxes)
        .sort((a, b) => a - b);
    }
    if (!verifyFull(level, moves)) return '';
    return moves;
  }

  // The level did not split into rooms: solve it whole.
  const cells: number[] = [];
  for (let i = 0; i < w * h; i++) if (wall[i] === 0) cells.push(i);
  const solved = solveRegion(level, cells, level.boxes, level.goals, level.player);
  if (solved === null) return '';
  if (!verifyFull(level, solved.moves)) return '';
  return solved.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}
