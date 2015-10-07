// A minimalistic Go-playing engine attempting to strike a balance between
// brevity, educational value and strength.
// Based on michi.py by Petr Baudis <pasky@ucw.cz> (https://github.com/pasky/michi)
package main

import (
    "fmt"
    "log"
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

func main() {
    log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
    log.Println("Start")

    fmt.Printf("%T %v\n", N, N)
    fmt.Printf("%T %v\n", W, W)
    fmt.Println(empty)

    fmt.Println(neighbors(23))
    fmt.Println(diag_neighbors(23))

    log.Println("End")
}
