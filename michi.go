// A minimalistic Go-playing engine attempting to strike a balance between
// brevity, educational value and strength.
// Based on michi.py by Petr Baudis <pasky@ucw.cz> (https://github.com/pasky/michi)
package main

import (
    "fmt"
    "log"
    "math/rand"
    "regexp"
    "strings"
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

// functions added to replace Python string methods
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

// create routine to replace random.shuffle() in Python
func Shuffle(a []int) {
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
        Shuffle(dlist)
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
func (p Position) score(owner_map []int) float32 {
    board := p.board
    var fboard string
    var touches_X, touches_x bool
    var komi float32
    var n int
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

    log.Println("End")
}
