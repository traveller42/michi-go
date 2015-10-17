// A minimalistic Go-playing engine attempting to strike a balance between
// brevity, educational value and strength.
// Based on michi.py by Petr Baudis <pasky@ucw.cz> (https://github.com/pasky/michi)
package main

import (
    "bufio"
    "fmt"
    "hash/fnv"
    "log"
    "math"
    "math/rand"
    "os"
    "regexp"
    "runtime"
    "strconv"
    "strings"
    "time"
)

// Given a board of size NxN (N=9, 19, ...), we represent the position
// as an (N+1)*(N+2) string, with '.' (empty), 'X' (to-play player),
// 'x' (other player), and whitespace (off-board border to make rules
// implementation easier).  Coordinates are just indices in this string.
// You can simply print(board) when debugging.
// "\n".join([(N+1)*' '] + N*[' '+N*'.'] + [(N+2)*' '])
const (
    N = 13
    W = N + 2
)
var empty = strings.Repeat(" ", N+1) + "\n" +
            strings.Repeat(" " + strings.Repeat(".", N) + "\n", N) +
            strings.Repeat(" ", N+2)
const (
    colstr = "ABCDEFGHJKLMNOPQRST"
    MAX_GAME_LEN = N * N * 3
)

const (
    N_SIMS = 1400
    RAVE_EQUIV = 3500
    EXPAND_VISITS = 8
    PRIOR_EVEN = 10  // should be even number; 0.5 prior
    PRIOR_SELFATARI = 10  // negative prior
    PRIOR_CAPTURE_ONE = 15
    PRIOR_CAPTURE_MANY = 30
    PRIOR_PAT3 = 10
    PRIOR_LARGEPATTERN = 100  // most moves have relatively small probability
)
var PRIOR_CFG = [...]int{24, 22, 8}  // priors for moves in cfg dist. 1, 2, 3
const (
    PRIOR_EMPTYAREA = 10
    REPORT_PERIOD = 200
)
var PROB_HEURISTIC = map[string]float32{
    "capture": 0.9,
    "pat3": 0.95,
    }  // probability of heuristic suggestions being taken in playout
const(
    PROB_SSAREJECT = 0.9  // probability of rejecting suggested self-atari in playout
    PROB_RSAREJECT = 0.5  // probability of rejecting random self-atari in playout; this is lower than above to allow nakade
    RESIGN_THRES = 0.2
    FASTPLAY20_THRES = 0.8  // if at 20% playouts winrate is >this, stop reading
    FASTPLAY5_THRES = 0.95  // if at 5% playouts winrate is >this, stop reading
)

var pat3src = [...][]string{  // 3x3 playout patterns; X,O are colors, x,o are their inverses
        {"XOX",  // hane pattern - enclosing hane
         "...",
         "???",},
        {"XO.",  // hane pattern - non-cutting hane
         "...",
         "?.?",},
        {"XO?",  // hane pattern - magari
         "X..",
         "x.?",},
     // {"XOO",  // hane pattern - thin hane
     //  "...",
     //  "?.?",} "X",  - only for the X player
        {".O.",  // generic pattern - katatsuke or diagonal attachment; similar to magari
         "X..",
         "...",},
        {"XO?",  // cut1 pattern (kiri] - unprotected cut
         "O.o",
         "?o?",},
        {"XO?",  // cut1 pattern (kiri] - peeped cut
         "O.X",
         "???",},
        {"?X?",  // cut2 pattern (de]
         "O.O",
         "ooo",},
        {"OX?",  // cut keima
         "o.O",
         "???",},
        {"X.?",  // side pattern - chase
         "O.?",
         "   ",},
        {"OX?",  // side pattern - block side cut
         "X.O",
         "   ",},
        {"?X?",  // side pattern - block side connection
         "x.O",
         "   ",},
        {"?XO",  // side pattern - sagari
         "x.x",
         "   ",},
        {"?OX",  // side pattern - cut
         "X.O",
         "   ",},
        }

var pat_gridcular_seq = [][][]int{  // Sequence of coordinate offsets of progressively wider diameters in gridcular metric
        {{0,0},
         {0,1}, {0,-1}, {1,0},  {-1,0},
         {1,1}, {-1,1}, {1,-1}, {-1,-1}, },  // d=1,2 is not considered separately
        {{0,2}, {0,-2}, {2,0},  {-2,0}, },
        {{1,2}, {-1,2}, {1,-2}, {-1,-2}, {2,1},  {-2,1},  {2,-1}, {-2,-1}, },
        {{0,3}, {0,-3}, {2,2},  {-2,2},  {2,-2}, {-2,-2}, {3,0},  {-3,0}, },
        {{1,3}, {-1,3}, {1,-3}, {-1,-3}, {3,1},  {-3,1},  {3,-1}, {-3,-1}, },
        {{0,4}, {0,-4}, {2,3},  {-2,3},  {2,-3}, {-2,-3}, {3,2},  {-3,2},  {3,-2}, {-3,-2}, {4,0},  {-4,0}, },
        {{1,4}, {-1,4}, {1,-4}, {-1,-4}, {3,3},  {-3,3},  {3,-3}, {-3,-3}, {4,1},  {-4,1},  {4,-1}, {-4,-1}, },
        {{0,5}, {0,-5}, {2,4},  {-2,4},  {2,-4}, {-2,-4}, {4,2},  {-4,2},  {4,-2}, {-4,-2}, {5,0},  {-5,0}, },
        {{1,5}, {-1,5}, {1,-5}, {-1,-5}, {3,4},  {-3,4},  {3,-4}, {-3,-4}, {4,3},  {-4,3},  {4,-3}, {-4,-3}, {5,1},  {-5,1},  {5,-1}, {-5,-1}, },
        {{0,6}, {0,-6}, {2,5},  {-2,5},  {2,-5}, {-2,-5}, {4,4},  {-4,4},  {4,-4}, {-4,-4}, {5,2},  {-5,2},  {5,-2}, {-5,-2}, {6,0},  {-6,0}, },
        {{1,6}, {-1,6}, {1,-6}, {-1,-6}, {3,5},  {-3,5},  {3,-5}, {-3,-5}, {5,3},  {-5,3},  {5,-3}, {-5,-3}, {6,1},  {-6,1},  {6,-1}, {-6,-1}, },
        {{0,7}, {0,-7}, {2,6},  {-2,6},  {2,-6}, {-2,-6}, {4,5},  {-4,5},  {4,-5}, {-4,-5}, {5,4},  {-5,4},  {5,-4}, {-5,-4}, {6,2},  {-6,2}, {6,-2}, {-6,-2}, {7,0}, {-7,0}, },
}
const (
    spat_patterndict_file = "patterns.spat"
    large_patterns_file = "patterns.prob"
)


//######################
// board string routines

// generator of coordinates for all neighbors of c
func neighbors(c int) []int {
    return []int{ c-1, c+1, c-W, c+W }
}

// generator of coordinates for all diagonal neighbors of c
func diag_neighbors(c int) []int {
    return []int{ c-W-1, c-W+1, c+W-1, c+W+1 }
}

func board_put(board string, c int, p string) string {
    return board[:c] + p + board[c+1:]
}

// replace continuous-color area starting at c with special color #
func floodfill(board string, c int) string {
    // XXX: Use bytearray to speed things up? (still needed in golang?)
    p := board[c]
    board = board_put(board, c, "#")
    fringe := []int{c}
    for len(fringe) > 0 {
        // c = fringe.pop()
        c, fringe = fringe[len(fringe)-1], fringe[:len(fringe)-1]
        // for d in neighbors(c)
        for _, d := range neighbors(c) {
            if board[d] == p {
                board = board_put(board, d, "#")
                // fringe.append(d)
                fringe = append(fringe, d)
            }
        }
    }
    return board
}

// Regex that matches various kind of points adjecent to '#' (floodfilled) points
func make_contact_res() map[string]*regexp.Regexp {
    temp_map := make(map[string]*regexp.Regexp)
    for _, p := range [...]string{".", "x", "X"} {
        rp := p
        if p == "." {
            rp = "\\."
        }
        contact_res_src := []string{
            "#" + rp, // p at right
            rp + "#", // p at left
            "#" + strings.Repeat(".", W-1) + rp, // p below
            rp + strings.Repeat(".", W-1) + "#", // p above
        }
        temp_map[p] = regexp.MustCompile("(?s:" + strings.Join(contact_res_src, "|") + ")")
    }
    return temp_map
}
var contact_res = make_contact_res()

// test if point of color p is adjecent to color # anywhere
// on the board; use in conjunction with floodfill for reachability
func contact(board, p string) int {
    // m = contact_res[p].search(board)
    m := contact_res[p].FindStringIndex(board)
    if m == nil {
        return -1
    }
    // return m.start() if m.group(0)[0] == p else m.end() - 1
    if board[m[0]:m[0]+1] == p {
        return m[0]
    }
    return m[1] - 1
}

// functions added to replace Python functions and methods

// use FNV Hash for Python hash function
func HashString(s string) uint64 {
    h := fnv.New64()
    h.Write([]byte(s))
    return h.Sum64()
}

func divmod(num, div int) (int, int) {
    return num / div, num % div
}

// mapping function to swap individual characters
func swapCase(r rune) rune {
    switch {
    case 'a' <= r && r <= 'z':
        return r - 'a' + 'A'
    case 'A' <= r && r <= 'Z':
        return r - 'A' + 'a'
    default:
        return r
    }
}
// function to apply mapping to string
func SwapCase(str string) string {
    return strings.Map(swapCase, str)
}

// create routines to replace random.shuffle() in Python
func ShuffleInt(a []int) {
    for i := range a {
        j := rand.Intn(i + 1)
        a[i], a[j] = a[j], a[i]
    }
}

func ShuffleTree(a []TreeNode) {
    for i := range a {
        j := rand.Intn(i + 1)
        a[i], a[j] = a[j], a[i]
    }
}

// create routine to test existence of element in slice
func intInSlice(intSlice []int, intTest int) bool {
    for _, i := range intSlice {
        if i == intTest {
            return true
        }
    }
    return false
}

func stringInSlice(strSlice []string, strTest string) bool {
    for _, str := range strSlice {
        if str == strTest {
            return true
        }
    }
    return false
}

// test for edge of board
func IsSpace(str string) bool {
    return strings.ContainsAny(str, " \n")
}

// test if c is inside a single-color diamond and return the diamond
// color or None; this could be an eye, but also a false one
func is_eyeish(board string, c int) string {
    eyecolor := ""
    othercolor := ""
    for _, d := range neighbors(c) {
        if IsSpace(board[d:d+1]) {
            continue
        }
        if board[d:d+1] == "." {
            return ""
        }
        if eyecolor == "" {
            eyecolor = board[d:d+1]
            othercolor = SwapCase(eyecolor)
        } else {
            if board[d:d+1] == othercolor {
                return ""
            }
        }
    }
    return eyecolor
}

// test if c is an eye and return its color or None
func is_eye(board string, c int) string {
    eyecolor := is_eyeish(board, c)
    if eyecolor == "" {
        return ""
    }

    // Eye-like shape, but it could be a falsified eye
    falsecolor := SwapCase(eyecolor)
    false_count := 0
    at_edge := false
    for _, d := range diag_neighbors(c) {
        if IsSpace(board[d:d+1]) {
            at_edge = true
        } else {
            if board[d:d+1] == falsecolor {
                false_count += 1
            }
        }
    }
    if at_edge {
        false_count += 1
    }
    if false_count >= 2 {
        return ""
    }
    return eyecolor
}

// class Position(namedtuple('Position', 'board cap n ko last last2 komi')):
// Implementation of simple Chinese Go rules
type Position struct {
    board string // string representation of board state
    cap []int    // holds total captured stones for each player
    n int        // n is how many moves were played so far
    ko int       // location of prohibited move under simple ko rules
    last int     // previous move
    last2 int    // antepenultimate move
    komi float32
}

// play as player X at the given coord c, return the new position
func (p Position) move(c int) (Position, string) {
    // Test for ko
    if c == p.ko {
        return p, "ko"
    }
    // Are we trying to play in enemy's eye?
    in_enemy_eye := is_eyeish(p.board, c) == "x"

    board := board_put(p.board, c, "X")
    // Test for captures, and track ko
    capX := p.cap[0]
    singlecaps := []int{}
    for _, d := range neighbors(c) {
        if board[d:d+1] != "x" {
            continue
        }
        // XXX: The following is an extremely naive and SLOW approach
        // at things - to do it properly, we should maintain some per-group
        // data structures tracking liberties.
        fboard := floodfill(board, d) // get a board with the adjacent group replaced by '#'
        if contact(fboard, ".") != -1 {
            continue  // some liberties left
        }
        // no liberties left for this group, remove the stones!
        capcount := strings.Count(fboard, "#")
        if capcount == 1 {
            singlecaps = append(singlecaps, d)
        }
        capX += capcount
        board = strings.Replace(fboard, "#", ".", -1) // capture the group
    }
    // set ko
    ko := -1
    if in_enemy_eye && len(singlecaps) == 1 {
        ko = singlecaps[0]
    }
    // Test for suicide
    if contact(floodfill(board, c), ".") == -1 {
        return p, "suicide"
    }

    // Update the position and return
    p.board = SwapCase(board)
    p.cap = []int{p.cap[1], capX}
    p.n = p.n + 1
    p.ko = ko
    p.last2 = p.last // must copy first
    p.last = c
    p.komi = p.komi

    return p, "ok"
}

// pass - i.e. return simply a flipped position
func (p Position) pass_move() (Position, string) {
    p.board = SwapCase(p.board)
    p.cap = []int{p.cap[1], p.cap[0]}
    p.n = p.n + 1
    p.ko = -1
    p.last2 = p.last // must copy first
    p.last = -1
    p.komi = p.komi

    return p, "ok"
}

// Generate a list of moves (includes false positives - suicide moves;
// does not include true-eye-filling moves), starting from a given board
// index (that can be used for randomization)
func (p Position) moves(i0 int) chan int {
    c := make(chan int)

    go func() {
        i := i0 - 1
        passes := 0
        for {
            i = strings.Index(p.board[i+1:], ".")
            if passes > 0 && (i==-1 || i >= i0) {
                close(c)
                break // we have looked through the whole board
            }
            if i == -1 {
                i = 0
                passes += 1
                continue // go back and start from the beginning
            }
            // Test for to-play player's one-point eye
            if is_eye(p.board, i) == "X" {
                continue
            }
            // yield i
            c <- i
        }

        close(c)
    }()
    return c
}

// generate a randomly shuffled list of points including and
// surrounding the last two moves (but with the last move having
// priority)
func (p Position) last_moves_neighbors() []int {
    clist := []int{}
    dlist := []int{}
    for _, c := range []int{p.last, p.last2} {
        if c == -1 {
            continue
        }
        // dlist = [c] + list(neighbors(c) + diag_neighbors(c))
        dlist = append([]int{c}, append(neighbors(c), diag_neighbors(c)...)...)
        ShuffleInt(dlist)
        // clist += [d for d in dlist if d not in clist]
        for _, d := range dlist {
            if intInSlice(clist, d) {
                continue
            }
            clist = append(clist, d)
        }
    }
    return clist
}

// compute score for to-play player; this assumes a final position
// with all dead stones captured; if owner_map is passed, it is assumed
// to be an array of statistics with average owner at the end of the game
// (+1 black, -1 white)
func (p Position) score(owner_map []float32) float32 {
    board := p.board
    var fboard string
    var touches_X, touches_x bool
    var komi float32
    var n float32
    i := 0
    for {
        i = strings.Index(p.board[i+1:], ".")
        if i == -1 {
            break
        }
        fboard = floodfill(board, i)
        // fboard is board with some continuous area of empty space replaced by #
        touches_X = contact(fboard, "X") != -1
        touches_x = contact(fboard, "x") != -1
        if touches_X && !touches_x {
            board = strings.Replace(fboard, "#", "X", -1)
        } else if touches_x && !touches_X {
            board = strings.Replace(fboard, "#", "x", -1)
        } else {
            board = strings.Replace(fboard, "#", ":", -1) // seki, rare
        }
    }
    // now that area is replaced either by X, x or :
    // komi = self.komi if self.n % 2 == 1 else -self.komi
    if p.n % 2 == 1 {
        komi = p.komi
    } else {
        komi = -p.komi
    }
    if len(owner_map) > 0 {
        for c := 0; c < W*W; c++ {
            if board[c:c+1] == "X" {
                n = 1
            } else if board[c:c+1] == "x" {
                n = -1
            } else {
                n = 0
            }
            if p.n % 2 == 1 {
                n = -n
            }
            owner_map[c] += n
        }
    }
    return float32(strings.Count(board, "X") - strings.Count(board, "x")) + komi
}

// Return an initial board position
func empty_position() Position {
    var p Position

    p.board = empty
    p.cap = []int{0, 0}
    p.n = 0
    p.ko = -1
    p.last = -1
    p.last2 = -1
    p.komi = 7.5

    return p
}

//##############
// go heuristics

// An atari/capture analysis routine that checks the group at c,
// determining whether (i) it is in atari (ii) if it can escape it,
// either by playing on its liberty or counter-capturing another group.
//  N.B. this is maybe the most complicated part of the whole program (sadly);
// feel free to just TREAT IT AS A BLACK-BOX, it's not really that
// interesting!
// The return value is a tuple of (boolean, [coord..]), indicating whether
// the group is in atari and how to escape/capture (or [] if impossible).
// (Note that (False, [...]) is possible in case the group can be captured
// in a ladder - it is not in atari but some capture attack/defense moves
// are available.)
// singlept_ok means that we will not try to save one-point groups;
// twolib_test means that we will check for 2-liberty groups which are
// threatened by a ladder
// twolib_edgeonly means that we will check the 2-liberty groups only
// at the board edge, allowing check of the most common short ladders
// even in the playouts
// def fix_atari(pos, c, singlept_ok=False, twolib_test=True, twolib_edgeonly=False):
func fix_atari(pos Position, c int, singlept_ok, twolib_test, twolib_edgeonly bool) (bool, []int) {

    // check if a capturable ladder is being pulled out at c and return
    // a move that continues it in that case; expects its two liberties as
    // l1, l2  (in fact, this is a general 2-lib capture exhaustive solver)
    read_ladder_attack := func(pos Position, c, l1, l2 int) int {
        for l := range([]int{l1, l2}) {
            pos_l, pos_err := pos.move(l)
            if pos_err != "ok" {
                continue
            }
            // fix_atari() will recursively call read_ladder_attack() back;
            // however, ignore 2lib groups as we don't have time to chase them
            // is_atari, atari_escape = fix_atari(pos_l, c, twolib_test=False)
            is_atari, atari_escape := fix_atari(pos_l, c, false, false, false)
            if is_atari && len(atari_escape) > 0 {
                return l
            }
        }
        return -1
    }

    fboard := floodfill(pos.board, c)
    group_size := strings.Count(fboard, "#")
    if singlept_ok && group_size == 1 {
        return false, []int{}
    }
    // Find a liberty
    l := contact(fboard, ".")
    // Ok, any other liberty?
    fboard = board_put(fboard, l, "L")
    l2 := contact(fboard, ".")
    if l2 != -1 {
        // At least two liberty group...
        if twolib_test && group_size > 1 &&
           (!twolib_edgeonly || line_height(l) == 0 && line_height(l2) == 0) &&
           contact(board_put(fboard, l2, "L"), ".") == -1 {
            // Exactly two liberty group with more than one stone.  Check
            // that it cannot be caught in a working ladder; if it can,
            // that's as good as in atari, a capture threat.
            // (Almost - N/A for countercaptures.)
            ladder_attack := read_ladder_attack(pos, c, l, l2)
            if ladder_attack >= 0 {
                return false, []int{ladder_attack}
            }
        }
        return false, []int{}
    }

    // In atari! If it's the opponent's group, that's enough...
    if pos.board[c:c+1] == "x" {
        return true, []int{l}
    }

    solutions := []int{}

    // Before thinking about defense, what about counter-capturing
    // a neighboring group?
    ccboard := fboard
    for {
        othergroup := contact(ccboard, "x")
        if othergroup == -1 {
            break
        }
        // a, ccls = fix_atari(pos, othergroup, twolib_test=False)
        a, ccls := fix_atari(pos, othergroup, false, false, false)
        if a && len(ccls) > 0 {
            solutions = append(solutions, ccls...)
        }
        // XXX: floodfill is better for big groups
        ccboard = board_put(ccboard, othergroup, "%")
    }

    // We are escaping.  Will playing our last liberty gain
    // at least two liberties?  Re-floodfill to account for connecting
    escpos, escerr := pos.move(l)
    if escerr != "ok" {
        return true, solutions // oops, suicidal move
    }
    fboard = floodfill(escpos.board, l)
    l_new := contact(fboard, ".")
    fboard = board_put(fboard, l_new, "L")
    l_new_2 := contact(fboard, ".")
    if l_new_2 != -1 {
        if len(solutions) > 0 ||
           !(contact(board_put(fboard, l_new_2, "L"), ".") == -1 &&
             read_ladder_attack(escpos, l, l_new, l_new_2) != -1) {
             solutions = append(solutions, l)
         }
    }
    return true, solutions
}

// return a board map listing common fate graph distances from
// a given point - this corresponds to the concept of locality while
// contracting groups to single points
func cfg_distance(board string, c int) []int {
    cfg_map := []int{}
    for i := 0; i < W*W; i++ {
        cfg_map = append(cfg_map, -1)
    }
    cfg_map[c] = 0

    // flood-fill like mechanics
    fringe := []int{c}
    for len(fringe) > 0 {
        // c = fringe.pop()
        c, fringe = fringe[len(fringe)-1], fringe[:len(fringe)-1]
        for _, d := range(neighbors(c)) {
            if IsSpace(board[d:d+1]) ||
               (0 <= cfg_map[d] && cfg_map[d] <= cfg_map[c]) {
                continue
            }
            cfg_before := cfg_map[d]
            if board[d:d+1] != "." && board[d:d+1] == board[c:c+1] {
                cfg_map[d] = cfg_map[c]
            } else {
                cfg_map[d] = cfg_map[c] + 1
            }
            if cfg_before < 0 || cfg_before > cfg_map[d] {
                fringe = append(fringe, d)
            }
        }
    }
    return cfg_map
}

// Return the line number above nearest board edge
func line_height(c int) int {
    row, col := divmod(c - (W+1), W)
    minVal := row
    for testVal := range([]int{col, N-1-row, N-1-col}) {
        if testVal < minVal {
            minVal = testVal
        }
    }
    return minVal
}

// Check whether there are any stones in Manhattan distance up to dist
// def empty_area(board, c, dist=3):
func empty_area(board string, c, dist int) bool {
    for d := range(neighbors(c)) {
        if strings.ContainsAny(board[d:d+1], "Xx") {
            return false
        }
        if board[d:d+1] == "." && dist > 1 && !empty_area(board, d, dist-1) {
            return false
        }
    }
    return true
}

// 3x3 pattern routines (those patterns stored in pat3src above)

// All possible neighborhood configurations matching a given pattern;
// used just for a combinatoric explosion when loading them in an
// in-memory set.
func pat3_expand(pat []string) []string {
    pat_rot90 := func(p []string) []string {
        return []string{p[2][0:1] + p[1][0:1] + p[2][0:1],
                        p[2][1:2] + p[1][1:2] + p[0][1:2],
                        p[2][2:3] + p[1][2:3] + p[0][2:3]}
    }
    pat_vertflip := func(p []string) []string {
        return []string{p[2], p[1], p[0]}
    }
    pat_horizflip := func(p []string) []string {
        return []string{p[0][2:3] + p[0][1:2] + p[0][0:1],
                        p[1][2:3] + p[1][1:2] + p[1][0:1],
                        p[2][2:3] + p[2][1:2] + p[2][0:1]}
    }
    pat_swapcolors := func(p []string) []string {
        l := []string{}
        for _, s:= range(p) {
            s = strings.Replace(s, "X", "Z", -1)
            s = strings.Replace(s, "x", "z", -1)
            s = strings.Replace(s, "O", "X", -1)
            s = strings.Replace(s, "o", "x", -1)
            s = strings.Replace(s, "Z", "O", -1)
            s = strings.Replace(s, "z", "o", -1)
            l = append(l, s)
        }
        return l
    }
    var pat_wildexp func(p, c, to string) []string
    pat_wildexp = func(p, c, to string) []string {
        i := strings.Index(p, c)
        if i == -1 {
            return []string{p}
        }
        l := []string{}
        for _, t := range(strings.Split(to, "")) {
            l = append(l, pat_wildexp(p[:i] + t + p[i+1:], c, to)...)
        }
        return l
    }
    pat_wildcards := func(pat string) []string {
        l := []string{}
        for _, p1 := range(pat_wildexp(pat, "?", ".XO ")) {
            for _, p2 := range(pat_wildexp(p1, "x", ".O ")) {
                for _, p3 := range(pat_wildexp(p2, "o", ".X ")) {
                    l = append(l, p3)
                }
            }
        }
        return l
    }

    rl := []string{}
    for _, p1 := range([][]string{pat, pat_rot90(pat)}) {
        for _, p2 := range([][]string{p1, pat_vertflip(p1)}) {
            for _, p3 := range([][]string{p2, pat_horizflip(p2)}) {
                for _, p4 := range([][]string{p3, pat_swapcolors(p3)}) {
                    for _, p5 := range(pat_wildcards(strings.Join(p4, ""))) {
                        rl = append(rl, p5)
                    }
                }
            }
        }
    }
    return rl
}

func pat3set_func() []string {
    l := []string{}
    for _, p := range(pat3src) {
        for _, s := range(pat3_expand(p)) {
            s = strings.Replace(s, "O", "x", -1)
            unique := true
            for _, t := range(l) {
                if s == t {
                    unique = false
                    break
                }
            }
            if unique {
                l = append(l, s)
            }
        }
    }
    return l
}
var pat3set = pat3set_func()

// return a string containing the 9 points forming 3x3 square around
//  certain move candidate
func neighborhood_33(board string, c int) string {
    return strings.Replace(board[c-W-1:c-W+2] + board[c-1:c+2] + board[c+W-1:c+W+2], "\n", " ", -1)
}

// large-scale pattern routines (those patterns living in patterns.{spat,prob} files)

// are you curious how these patterns look in practice? get
// https://github.com/pasky/pachi/blob/master/tools/pattern_spatial_show.pl
// and try e.g. ./pattern_spatial_show.pl 71

var spat_patterndict = make(map[uint64]int) // hash(neighborhood_gridcular()) -> spatial id

// load dictionary of positions, translating them to numeric ids
func load_spat_patterndict(f *os.File) {
    scanner := bufio.NewScanner(f)
    for scanner.Scan() {
        line := scanner.Text()
        // line: 71 6 ..X.X..OO.O..........#X...... 33408f5e 188e9d3e 2166befe aa8ac9e 127e583e 1282462e 5e3d7fe 51fc9ee
        if strings.HasPrefix(line, "#") {
            continue
        }
        neighborhood := strings.Replace(strings.Replace(strings.Split(line, " ")[2], "#", " ", -1), "O", "x", -1)
        if id, err := strconv.ParseInt(strings.Split(line, " ")[0], 10, 0); err == nil {
            spat_patterndict[HashString(neighborhood)] = int(id)
        }
    }
}

var large_patterns = make(map[int]float32) // spatial id -> probability

// dictionary of numeric pattern ids, translating them to probabilities
// that a move matching such move will be played when it is available
func load_large_patterns(f *os.File) {
    re := regexp.MustCompile("s:([0-9]+)")
    scanner := bufio.NewScanner(f)
    for scanner.Scan() {
        line := scanner.Text()
        // line: 0.004 14 3842 (capture:17 border:0 s:784)
        if p, err := strconv.ParseFloat(strings.Split(line, " ")[0],32); err == nil {
            if m := re.FindStringSubmatch(line); m != nil {
                if s, err := strconv.ParseInt(m[1], 10, 0); err == nil {
                    large_patterns[int(s)] = float32(p)
                }
            }
        }
    }
}

// Yield progressively wider-diameter gridcular board neighborhood
// stone configuration strings, in all possible rotations
func neighborhood_gridcular(board string, c int, done chan bool) chan string {
    ch := make(chan string)

    go func() {
        // Each rotations element is (xyindex, xymultiplier)
        rotations := [][][]int{
            {{0,1},{1,1}},
            {{0,1},{-1,1}},
            {{0,1},{1,-1}},
            {{0,1},{-1,-1}},
            {{1,0},{1,1}},
            {{1,0},{-1,1}},
            {{1,0},{1,-1}},
            {{1,0},{-1,-1}},
        }
        wboard := strings.Replace(board, "\n", " ", -1)
        for _, dseq := range(pat_gridcular_seq) {
            for ri := 0; ri < len(rotations); ri++ {
                r := rotations[ri]
                neighborhood := ""
                for _, o := range(dseq) {
                    y, x := divmod(c - (W+1), W)
                    y += o[r[0][0]]*r[1][0]
                    x += o[r[0][1]]*r[1][1]
                    if y >= 0 && y < N && x >= 0 && x < N {
                        si := (y+1)*W + x+1
                        neighborhood += wboard[si:si+1]
                    } else {
                        neighborhood += " "
                    }
                }
                select {
                    case ch <- neighborhood:
                    case <-done:
                        close(ch)
                        return
                }
            }
        }

        close(ch)
    }()
    return ch
}

// return probability of large-scale pattern at coordinate c.
// Multiple progressively wider patterns may match a single coordinate,
// we consider the largest one.
func large_pattern_probability(board string, c int) float32 {
    probability := float32(-1)
    matched_len := 0
    non_matched_len := 0
    done := make(chan bool)
    for n := range(neighborhood_gridcular(board, c, done)) {
        sp_i, good_sp_i := spat_patterndict[HashString(n)]
        if good_sp_i {
            prob, good_prob := large_patterns[sp_i]
            if good_prob {
                probability = prob
                matched_len = len(n)
                continue
            }
        }
        if matched_len < non_matched_len && non_matched_len < len(n) {
            // stop when we did not match any pattern with a certain
            // diameter - it ain't going to get any better!
            done <- true
            break
        }
        non_matched_len = len(n)
    }
    close(done)
    return probability
}

//##########################
// montecarlo playout policy

// Yield candidate next moves in the order of preference; this is one
// of the main places where heuristics dwell, try adding more!
//
// heuristic_set is the set of coordinates considered for applying heuristics;
// this is the immediate neighborhood of last two moves in the playout, but
// the whole board while prioring the tree.
// def gen_playout_moves(pos, heuristic_set, probs={'capture': 1, 'pat3': 1}, expensive_ok=False):
type Result struct { intResult int
                     strResult string}

func gen_playout_moves(pos Position, heuristic_set []int, probs map[string]float32, expensive_ok bool) chan Result {
    ch := make(chan Result)
    var r Result

    go func() {
        // Check whether any local group is in atari and fill that liberty
        // print('local moves', [str_coord(c) for c in heuristic_set], file=sys.stderr)
        if rand.Float32() <= probs["capture"] {
            already_suggested := []int{}
            for _, c := range(heuristic_set) {
                if strings.ContainsAny(pos.board[c:c+1], "Xx") {
                    // in_atari, ds = fix_atari(pos, c, twolib_edgeonly=not expensive_ok)
                    _, ds := fix_atari(pos, c, false, true, !(expensive_ok))
                    ShuffleInt(ds)
                    for _, d := range(ds) {
                        if !(intInSlice(already_suggested, d)) {
                            r.intResult = d
                            r.strResult = "capture " + strconv.FormatInt(int64(c), 10)
                            ch <- r
                            already_suggested = append(already_suggested, d)
                        }
                    }
                }
            }
        }

        // Try to apply a 3x3 pattern on the local neighborhood
        if rand.Float32() <= probs["pat3"] {
            already_suggested := []int{}
            for _, c := range(heuristic_set) {
                if pos.board[c:c+1] == "." && !(intInSlice(already_suggested, c)) && stringInSlice(pat3set, neighborhood_33(pos.board, c)) {
                    r.intResult = c
                    r.strResult = "pat3"
                    ch <- r
                    already_suggested = append(already_suggested, c)
                }
            }
        }

        // Try *all* available moves, but starting from a random point
        // (in other words, suggest a random move)
        x, y := rand.Intn(N-1)+1, rand.Intn(N-1)+1
        for c := range(pos.moves(y*W + x)) {
            r.intResult = c
            r.strResult = "random"
            ch <- r
        }

        close(ch)
    }()
    return ch
}

// Start a Monte Carlo playout from a given position,
// return score for to-play player at the starting position;
// amaf_map is board-sized scratchpad recording who played at a given
// position first
// def mcplayout(pos, amaf_map, disp=False):
func mcplayout(pos Position, amaf_map []int, disp bool) (float32, []int, []float32) {
    var pos2 Position
    var prob_reject float32
    if disp {
        fmt.Println(os.Stderr, "** SIMULATION **")
    }
    start_n := pos.n
    passes := 0
    for passes < 2 && pos.n < MAX_GAME_LEN {
        if disp {
            print_pos(pos, os.Stderr, nil)
        }
        pos2.n = -99
        // We simply try the moves our heuristics generate, in a particular
        // order, but not with 100% probability; this is on the border between
        // "rule-based playouts" and "probability distribution playouts".
        for r := range(gen_playout_moves(pos, pos.last_moves_neighbors(), PROB_HEURISTIC, false)) {
            c := r.intResult
            kind := r.strResult
            if disp && kind != "random" {
                fmt.Println(os.Stderr, "move suggestion", str_coord(c), kind)
            }
            pos2, err := pos.move(c)
            if err != "ok" {
                continue
            }
            // check if the suggested move did not turn out to be a self-atari
            if kind == "random" {
                prob_reject = PROB_RSAREJECT
            } else {
                prob_reject = PROB_SSAREJECT
            }
            if rand.Float32() <= prob_reject {
                // in_atari, ds = fix_atari(pos2, c, singlept_ok=True, twolib_edgeonly=True)
                _, ds := fix_atari(pos2, c, true, true, true)
                if len(ds) > 0 {
                    if disp {
                        fmt.Println(os.Stderr, "rejecting self-atari move", str_coord(c))
                    }
                    pos2.n = -99
                    continue
                }
            }
            if amaf_map[c] == 0 { // Mark the coordinate with 1 for black
                if pos.n % 2 == 0 {
                    amaf_map[c] = 1
                } else {
                    amaf_map[c] = -1
                }
            }
            break
        }
        if pos2.n == -99 { // no valid moves, pass
            pos, _ = pos.pass_move()
            passes += 1
            continue
        }
        passes = 0
        pos = pos2
    }
    owner_map := make([]float32, W*W)
    score := pos.score(owner_map)
    if disp {
        if pos.n % 2 == 0 {
            fmt.Fprintf(os.Stderr, "** SCORE B%+.1f **\n", score)
        } else {
            fmt.Fprintf(os.Stderr, "** SCORE B%+.1f **\n", -score)
        }
    }
    if start_n % 2 != pos.n % 2 {
        score = -score
    }
    return score, amaf_map, owner_map
}

//#######################
// montecarlo tree search

// Monte-Carlo tree node;
// v is #visits, w is #wins for to-play (expected reward is w/v)
// pv, pw are prior values (node value = w/v + pw/pv)
// av, aw are amaf values ("all moves as first", used for the RAVE tree policy)
// children is None for leaf nodes
type TreeNode struct {
    pos Position
    v int   // # of visits
    w int   // # wins
    pv int  // prior value of v
    pw int  // prior value of w
    av int
    aw int
    children []TreeNode
}

func NewTreeNode(pos Position) TreeNode {
    var tn TreeNode
    tn.pos = pos
    tn.v = 0
    tn.w = 0
    tn.pv = PRIOR_EVEN
    tn.pw = PRIOR_EVEN/2
    tn.av = 0
    tn.aw = 0
    tn.children = []TreeNode{}
    return tn
}

// add and initialize children to a leaf node
func (tn TreeNode) expand() {
    cfg_map := []int{}
    if tn.pos.last != -1 {
        cfg_map = append(cfg_map, cfg_distance(tn.pos.board, tn.pos.last)...)
    }
    tn.children = []TreeNode{}
    childset := map[int]TreeNode{}
    // Use playout generator to generate children and initialize them
    // with some priors to bias search towards more sensible moves.
    // Note that there can be many ways to incorporate the priors in
    // next node selection (progressive bias, progressive widening, ...).
    seed_set := []int{}
    for i := N; i < (N+1)*W; i++ {
        seed_set = append(seed_set, i)
    }
    for r:= range(gen_playout_moves(tn.pos, seed_set, map[string]float32{"capture": 1, "pat3": 1}, true)) {
        c := r.intResult
        kind := r.strResult
        pos2, err := tn.pos.move(c)
        if err != "ok" {
            continue
        }
        // n_playout_moves() will generate duplicate suggestions
        // if a move is yielded by multiple heuristics
        node, ok := childset[pos2.last]
        if !ok {
            node = NewTreeNode(pos2)
            tn.children = append(tn.children, node)
            childset[pos2.last] = node
        }

        if strings.HasPrefix(kind, "capture") {
            // Check how big group we are capturing; coord of the group is
            // second word in the ``kind`` string
            coord, _ := strconv.ParseInt(strings.Split(kind, " ")[1], 10, 32)
            if strings.Count(floodfill(tn.pos.board, int(coord)), "#") > 1 {
                node.pv += PRIOR_CAPTURE_MANY
                node.pw += PRIOR_CAPTURE_MANY
            } else {
                node.pv += PRIOR_CAPTURE_ONE
                node.pw += PRIOR_CAPTURE_ONE
            }
        } else if kind == "pat3" {
            node.pv += PRIOR_PAT3
            node.pw += PRIOR_PAT3
        }
    }

    // Second pass setting priors, considering each move just once now
    for _, node := range(tn.children) {
        c := node.pos.last

        if len(cfg_map) > 0 && cfg_map[c]-1 < len(PRIOR_CFG) {
            node.pv += PRIOR_CFG[cfg_map[c]-1]
            node.pw += PRIOR_CFG[cfg_map[c]-1]
        }

        height := line_height(c) // 0-indexed
        // if height <= 2 and empty_area(self.pos.board, c):
        if height <= 2 && empty_area(tn.pos.board, c, 3) {
            // No stones around; negative prior for 1st + 2nd line, positive
            // for 3rd line; sanitizes opening and invasions
            if height <= 1 {
                node.pv += PRIOR_EMPTYAREA
                node.pw += 0
            }
            if height == 2 {
                node.pv += PRIOR_EMPTYAREA
                node.pw += PRIOR_EMPTYAREA
            }
        }

        // in_atari, ds = fix_atari(node.pos, c, singlept_ok=True)
        _, ds := fix_atari(node.pos, c, true, true, false)
        if len(ds) > 0 {
            node.pv += PRIOR_SELFATARI
            node.pw += 0 // negative prior
        }

        patternprob := large_pattern_probability(tn.pos.board, c)
        if patternprob > 0.001 {
            pattern_prior := float32(math.Sqrt(float64(patternprob))) // tone up
            node.pv += int(pattern_prior * PRIOR_LARGEPATTERN)
            node.pw += int(pattern_prior * PRIOR_LARGEPATTERN)
        }
    }

    if len(tn.children) == 0 {
        // No possible moves, add a pass move
        pass_pos, _ := tn.pos.pass_move()
        tn.children = append(tn.children, NewTreeNode(pass_pos))
    }
}

func (tn TreeNode) rave_urgency() float32 {
    v := tn.v + tn.pv
    expectation := float32(tn.w + tn.pw) / float32(v)
    if tn.av == 0 {
        return expectation
    }
    rave_expectation := float32(tn.aw) / float32(tn.av)
    beta := float32(tn.av) / (float32(tn.av + v) + float32(v) * float32(tn.av) / RAVE_EQUIV)
    return beta * rave_expectation + (1-beta) * expectation
}

func (tn TreeNode) winrate() float32 {
    if tn.v > 0 {
        return float32(tn.w) / float32(tn.v)
    } else {
        return float32(math.NaN())
    }
}

// best move is the most simulated one
func (tn TreeNode) best_move() (TreeNode, bool) {
    var max_node TreeNode
    if len(tn.children) == 0 {
        return NewTreeNode(empty_position()), false
    } else {
        max_v := -1
        for _, node := range(tn.children) {
            if node.v > max_v {
                max_v = node.v
                max_node = node
            }
        }
    }
    return max_node, true
}

// Descend through the tree to a leaf
func tree_descend(tree TreeNode, amaf_map []int, disp bool) []TreeNode {
    tree.v += 1
    nodes := []TreeNode{tree}
    passes := 0
    for len(nodes[len(nodes)-1].children) > 0 && passes < 2 {
        if disp {
            print_pos(nodes[len(nodes)-1].pos, os.Stderr, []float32{})
        }

        // Pick the most urgent child
        children := nodes[len(nodes)-1].children
        if disp {
            for _, c := range(children) {
                // dump_subtree(c, recurse=False)
                dump_subtree(c, N_SIMS/50, 0, os.Stderr, false)
            }
        }
        ShuffleTree(children) // randomize the max in case of equal urgency

        // node = max(children, key=lambda node: node.rave_urgency())
        node := children[0]
        max_rave := node.rave_urgency()
        for _, c := range(children) {
            test_rave := c.rave_urgency()
            if test_rave > max_rave {
                node = c
                max_rave = test_rave
            }
        }
        nodes = append(nodes, node)

        if disp {
            fmt.Fprintf(os.Stderr, "chosen %s\n", str_coord(node.pos.last))
        }
        if node.pos.last == -1 {
            passes += 1
        } else {
            passes = 0
            if amaf_map[node.pos.last] == 0 { // Mark the coordinate with 1 for black
                // amaf_map[node.pos.last] = 1 if nodes[-2].pos.n % 2 == 0 else -1
                if nodes[len(nodes)-2].pos.n % 2 == 0 {
                    amaf_map[node.pos.last] = 1
                } else {
                    amaf_map[node.pos.last] = -1
                }
            }
        }
        // updating visits on the way *down* represents "virtual loss", relevant for parallelization
        node.v += 1
        if len(node.children) == 0 && node.v >= EXPAND_VISITS {
            node.expand()
        }
    }
    return nodes
}

// Store simulation result in the tree (@nodes is the tree path)
// def tree_update(nodes, amaf_map, score, disp=False):
func tree_update(nodes []TreeNode, amaf_map []int, score float32, disp bool) {
    local_nodes := []TreeNode{}
    limit := len(nodes)
    for i := 1; i <= limit; i++ {
        local_nodes = append(local_nodes, nodes[limit-i])
    }
    for _, node := range(local_nodes) {
        if disp {
            fmt.Println(os.Stderr, "updating", str_coord(node.pos.last), score < 0)
        }
        win := 0
        if score < 0 { // score is for to-play, node statistics for just-played
            win = 1
        }
        node.w += win
        // Update the node children AMAF stats with moves we made
        // with their color
        // amaf_map_value = 1 if node.pos.n % 2 == 0 else -1
        amaf_map_value := 1
        if node.pos.n % 2 != 0 {
            amaf_map_value = -1
        }
        if len(node.children) > 0 {
            for _, child := range(node.children) {
                if child.pos.last == -1 {
                    continue
                }
                if amaf_map[child.pos.last] == amaf_map_value {
                    if disp {
                        fmt.Println(os.Stderr, "  AMAF updating", str_coord(child.pos.last), score > 0)
                    }
                    // child.aw += score > 0
                    win = 0
                    if score > 0 { // reversed perspective
                        win = 1
                    }
                    child.aw += win
                    child.av += 1
                }
            }
        }
        score = -score
    }
}

// In original Python (not used in the Go translation)
// worker_pool = None

// Perform MCTS search from a given position for a given #iterations
// def tree_search(tree, n, owner_map, disp=False):
func tree_search(tree TreeNode, n int, owner_map []float32, disp bool) TreeNode {
    // Initialize root node
    if len(tree.children) == 0 {
        tree.expand()
    }

    // We could simply run tree_descend(), mcplayout(), tree_update()
    // sequentially in a loop.  This is essentially what the code below
    // does, if it seems confusing!

    // However, we also have an easy (though not optimal) way to parallelize
    // by distributing the mcplayout() calls to other processes using the
    // multiprocessing Python module.  mcplayout() consumes maybe more than
    // 90% CPU, especially on larger boards.  (Except that with large patterns,
    // expand() in the tree descent phase may be quite expensive - we can tune
    // that tradeoff by adjusting the EXPAND_VISITS constant.)

    n_workers := runtime.NumCPU()
    if disp { // set to 1 when debugging
        n_workers = 1
    }

    // global worker_pool
    // if worker_pool is None:
    //   worker_pool = Pool(processes=n_workers)

    type Job struct {
        nodes []TreeNode
        amaf_map []int
        owner_map []float32
        score float32
    }
    type JobResult struct {
        n int
        job Job
    }
    jr := make(chan JobResult)
    outgoing := []Job{}      // positions waiting for a playout
    ongoing := map[int]Job{} // currently ongoing playout jobs
    incoming := []Job{}      //positions that finished evaluation
    i := 0
    for i < n {
        if len(outgoing) == 0 && !(disp && len(ongoing) > 0) {
            // Descend the tree so that we have something ready when a worker
            // stops being busy
            amaf_map := make([]int, W*W)
            nodes := tree_descend(tree, amaf_map, disp)
            var job Job
            job.nodes = nodes
            job.amaf_map = amaf_map
            outgoing = append(outgoing, job)
        }

        if len(ongoing) >= n_workers {
            // Too many playouts running? Wait a bit...
            time.Sleep(10 * time.Millisecond / time.Duration(n_workers))
        } else {
            i += 1
            if i > 0 && i % REPORT_PERIOD == 0 {
                print_tree_summary(tree, i, os.Stderr)
            }

            // Issue an mcplayout job to the worker pool
            // nodes, amaf_map = outgoing.pop()
            dispatch := outgoing[len(outgoing)-1]
            outgoing = outgoing[:len(outgoing)-1]
            jobnum := i
            go func() {
                var jobresult JobResult
                jobresult.n = jobnum
                jobresult.job = dispatch
                jobresult.job.score, jobresult.job.amaf_map, jobresult.job.owner_map = mcplayout(jobresult.job.nodes[len(jobresult.job.nodes)-1].pos, jobresult.job.amaf_map, disp)
                jr <- jobresult
            }()
            ongoing[jobnum] = dispatch
        }

        // Anything to store in the tree?  (We do this step out-of-order
        // picking up data from the previous round so that we don't stall
        // ready workers while we update the tree.)
        for len(incoming) > 0 {
            result := incoming[len(incoming)-1]
            incoming = incoming[:len(incoming)-1]
            tree_update(result.nodes, result.amaf_map, result.score, disp)
            for c := 0; c < W*W; c++ {
                owner_map[c] = result.owner_map[c]
            }
        }

        // Any playouts are finished yet?
        select {
        case jobresult := <- jr: // Yes! Queue them up for storing in the tree.
            incoming = append(incoming, jobresult.job)
            delete(ongoing, jobresult.n)
        default:
        }

        // Early stop test
        best_move, ok := tree.best_move()
        if ok {
            best_wr:= best_move.winrate()
            if (i > n/20 && best_wr > FASTPLAY5_THRES) || (i > n/5 && best_wr > FASTPLAY20_THRES) {
                break
            }
        }
    }
    close(jr)

    for c:= 0; c < W*W; c++ {
        owner_map[c] = owner_map[c] / float32(i)
    }
    dump_subtree(tree, N_SIMS/50, 0, os.Stderr, true)
    print_tree_summary(tree, i, os.Stderr)
    best_move, _ := tree.best_move()
    return best_move
}

//##################
// user interface(s)

// utility routines

// print visualization of the given board position, optionally also
// including an owner map statistic (probability of that area of board
// eventually becoming black/white)
// def print_pos(pos, f=sys.stderr, owner_map=None):
func print_pos(pos Position, f *os.File, owner_map []float32) {
    var Xcap, Ocap int
    var board string
    if pos.n % 2 == 0 { // to-play is black
        board = strings.Replace(pos.board, "x", "O", -1)
        Xcap, Ocap = pos.cap[0], pos.cap[1]
    } else { // to-play is white
        board = strings.Replace(strings.Replace(pos.board, "X", "O", -1), "x", "X", -1)
        Ocap, Xcap = pos.cap[0], pos.cap[1]
    }
    fmt.Fprintf(f, "Move: %-3d   Black: %d caps   White: %d caps   Komi: %.1f\n", pos.n, Xcap, Ocap, pos.komi)
    pretty_board := strings.TrimRight(board, " \n") + " "
    if pos.last != -1 {
        pretty_board = pretty_board[:pos.last*2-1] + "(" + board[pos.last:pos.last+1] + ")" + pretty_board[pos.last*2+2:]
    }
    pb := []string{}
    for i, row := range(strings.Split(pretty_board, "\n")[1:]) {
        row = fmt.Sprintf(" %-02d%s", N-i, row[2:])
        pb = append(pb, row)
    }
    pretty_board = strings.Join(pb, "\n")
    if len(owner_map) > 0 {
        pretty_ownermap := ""
        for c := 0; c < W*W; c++ {
            if IsSpace(board[c:c+1]) {
                pretty_ownermap += board[c:c+1]
            } else if owner_map[c] > 0.6 {
                pretty_ownermap += "X"
            } else if owner_map[c] > 0.3 {
                pretty_ownermap += "x"
            } else if owner_map[c] < -0.6 {
                pretty_ownermap += "O"
            } else if owner_map[c] < -0.3 {
                pretty_ownermap += "o"
            } else {
                pretty_ownermap += "."
            }
        }
        pretty_ownermap = strings.TrimRight(pretty_ownermap, " \n")
        pb2 := []string{}
        for i, orow := range(strings.Split(pretty_ownermap, "\n")[1:]) {
            row := fmt.Sprintf("%s  %s", pb[i-1], orow[2:])
            pb2 = append(pb2, row)
        }
        pretty_board = strings.Join(pb2, "\n")
    }
    fmt.Println(f, pretty_board)
    fmt.Println(f, "    " + colstr[:N])
    fmt.Println(f, "")
}

// Sort a slice of TreeNode by the v field
// Return with Max v
func Best_Nodes(nodes []TreeNode) []TreeNode {
    var i_max, v_max int

    if len(nodes) == 1 {
        return nodes
    }
    i_max = -1
    v_max = -1
    for i, node := range(nodes) {
        if node.v > v_max {
            i_max = i
            v_max = node.v
        }
    }
    best_nodes := []TreeNode{nodes[i_max]}
    remaining_nodes := nodes[:i_max]
    if i_max < len(nodes)-1 {
        remaining_nodes = append(remaining_nodes, nodes[i_max+1:]...)
    }
    return append(best_nodes, Best_Nodes(remaining_nodes)...)
}

// print this node and all its children with v >= thres.
// def dump_subtree(node, thres=N_SIMS/50, indent=0, f=sys.stderr, recurse=True):
func dump_subtree(node TreeNode, thres, indent int, f *os.File, recurse bool) {
    var float_val float32
    if node.av > 0 {
        float_val = float32(node.aw)/float32(node.av)
    } else{
        float_val = float32(math.NaN())
    }
    fmt.Fprintf(f, "%s+- %s %.3f (%d/%d, prior %d/%d, rave %d/%d=%.3f, urgency %.3f)\n",
                strings.Repeat(" ", indent), str_coord(node.pos.last), node.winrate(),
                node.w, node.v, node.pw, node.pv, node.aw, node.av, float_val,
                node.rave_urgency())
    if !recurse {
        return
    }
    children := node.children
    for _, child := range(Best_Nodes(children)) {
        if child.v >= thres {
            dump_subtree(child, thres, indent+3, f, true)
        }
    }
}

// def print_tree_summary(tree, sims, f=sys.stderr):
func print_tree_summary(tree TreeNode, sims int, f *os.File) {
    var exists bool
    best_nodes := Best_Nodes(tree.children)[:5]
    best_seq := []int{}
    node := tree
    for {
        best_seq = append(best_seq, node.pos.last)
        node, exists = node.best_move()
        if !exists { // no children of current node
            break
        }
    }
    seq_string := ""
    for _, c := range(best_seq[1:6]) {
        seq_string += str_coord(c) + " "
    }
    best_nodes_string := ""
    for _, n := range(best_nodes) {
        best_nodes_string += fmt.Sprintf("%s(%.3f) ", str_coord(n.pos.last), n.winrate())
    }
    fmt.Fprintf(f, "[%4d] winrate %.3f | seq %s | can %s", sims, best_nodes[0].winrate(),
                seq_string, best_nodes_string)
}

func parse_coord(s string) int {
    if s == "pass" {
        return -1
    }
    row, _ := strconv.ParseInt(s[1:], 10, 32)
    return W+1 + (N - int(row) * W + strings.Index(colstr, strings.ToUpper(s[0:1])))
}

func str_coord(c int) string {
    if c == -1 {
        return "pass"
    }
    row, col := divmod(c - (W+1), W)
    return fmt.Sprintf("%c%d", colstr[col], N-row)
}

// various main programs

// run n Monte-Carlo playouts from empty position, return avg. score
func mcbenchmark(n int) float32 {
    var sumscore float32
    for i := 0; i < n; i++ {
        score, _, _ := mcplayout(empty_position(), make([]int, W*W), false)
        sumscore += score
    }
    return sumscore / float32(n)
}

func main() {
    log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
    log.Println("Start")

    fmt.Printf("%T %v\n", N, N)
    fmt.Printf("%T %v\n", W, W)
    fmt.Println(empty)

    fmt.Println(neighbors(23))
    fmt.Println(diag_neighbors(23))

    p := empty_position()
    fmt.Println(p)

    fmt.Println(len(pat3set))

    log.Println("MC Test Start")
    fmt.Println(mcbenchmark(100))
    log.Println("MC Test End")

    log.Println("End")
}
