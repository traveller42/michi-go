// A minimalistic Go-playing engine attempting to strike a balance between
// brevity, educational value and strength.
// Based on michi.py by Petr Baudis <pasky@ucw.cz> (https://github.com/pasky/michi)
package main

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode"

	xxhash "github.com/OneOfOne/xxhash"
	mt64 "github.com/bszcz/mt19937_64"
)

// Given a board of size NxN (N=9, 19, ...), we represent the position
// as an (N+1)*(N+2) string, with '.' (empty), 'X' (to-play player),
// 'x' (other player), and whitespace (off-board border to make rules
// implementation easier).  Coordinates are just indices in this string.
// You can simply print(board) when debugging.
const (
	N            = 13                    // boardsize
	W            = N + 2                 // arraysize including buffer for board edge
	columnString = "ABCDEFGHJKLMNOPQRST" // labels for columns
	MAX_GAME_LEN = N * N * 3
)

// emptyBoard is a byte slice representing an empty board
var emptyBoard []byte

const (
	PASS = -1346458457 // 'P','A','S','S' 0x50415353
	NONE = -1313820229 // 'N','O','N','E' 0x4e4f4e45
)

// constants related to the operation of the MCTS move selection
const (
	N_SIMS             = 1400 // number of playouts for Monte-Carlo Search
	RAVE_EQUIV         = 3500
	EXPAND_VISITS      = 8
	PRIOR_EVEN         = 10 // should be even number; 0.5 prior
	PRIOR_SELFATARI    = 10 // negative prior
	PRIOR_CAPTURE_ONE  = 15
	PRIOR_CAPTURE_MANY = 30
	PRIOR_PAT3         = 10
	PRIOR_LARGEPATTERN = 100 // most moves have relatively small probability
	PRIOR_EMPTYAREA    = 10
	REPORT_PERIOD      = 200
	PROB_SSAREJECT     = 0.9 // probability of rejecting suggested self-atari in playout
	PROB_RSAREJECT     = 0.5 // probability of rejecting random self-atari in playout; this is lower than above to allow nakade
	RESIGN_THRES       = 0.2
	FASTPLAY20_THRES   = 0.8  // if at 20% playouts winrate is >this, stop reading
	FASTPLAY5_THRES    = 0.95 // if at 5% playouts winrate is >this, stop reading
)

var PRIOR_CFG = [...]int{24, 22, 8} // priors for moves in cfg dist. 1, 2, 3

// probability of heuristic suggestions being taken in playout
var PROB_HEURISTIC = map[string]float64{
	"capture": 0.9,
	"pat3":    0.95,
}

// filenames for the large patterns from the Pachi Go engine
const (
	spatPatternDictFile = "patterns.spat"
	largePatternsFile   = "patterns.prob"
)

// 3x3 playout patterns; X,O are colors, x,o are their inverses
var pattern3x3Source = [...][][]byte{
	{{'X', 'O', 'X'}, // hane pattern - enclosing hane
		{'.', '.', '.'},
		{'?', '?', '?'}},
	{{'X', 'O', '.'}, // hane pattern - non-cutting hane
		{'.', '.', '.'},
		{'?', '.', '?'}},
	{{'X', 'O', '?'}, // hane pattern - magari
		{'X', '.', '.'},
		{'x', '.', '?'}},
	// {{'X','O','O'},  // hane pattern - thin hane
	//  {'.','.','.'},
	//  {'?','.','?'}} "X",  - only for the X player
	{{'.', 'O', '.'}, // generic pattern - katatsuke or diagonal attachment; similar to magari
		{'X', '.', '.'},
		{'.', '.', '.'}},
	{{'X', 'O', '?'}, // cut1 pattern (kiri] - unprotected cut
		{'O', '.', 'o'},
		{'?', 'o', '?'}},
	{{'X', 'O', '?'}, // cut1 pattern (kiri] - peeped cut
		{'O', '.', 'X'},
		{'?', '?', '?'}},
	{{'?', 'X', '?'}, // cut2 pattern (de]
		{'O', '.', 'O'},
		{'o', 'o', 'o'}},
	{{'O', 'X', '?'}, // cut keima
		{'o', '.', 'O'},
		{'?', '?', '?'}},
	{{'X', '.', '?'}, // side pattern - chase
		{'O', '.', '?'},
		{' ', ' ', ' '}},
	{{'O', 'X', '?'}, // side pattern - block side cut
		{'X', '.', 'O'},
		{' ', ' ', ' '}},
	{{'?', 'X', '?'}, // side pattern - block side connection
		{'x', '.', 'O'},
		{' ', ' ', ' '}},
	{{'?', 'X', 'O'}, // side pattern - sagari
		{'x', '.', 'x'},
		{' ', ' ', ' '}},
	{{'?', 'O', 'X'}, // side pattern - cut
		{'X', '.', 'O'},
		{' ', ' ', ' '}},
}

// 3x3 pattern routines (those patterns stored in pattern3x3Source above)

// All possible neighborhood configurations matching a given pattern;
// used just for a combinatoric explosion when loading them in an
// in-memory set.
func pat3_expand(pat [][]byte) [][]byte {
	pat_rot90 := func(p [][]byte) [][]byte {
		return [][]byte{{p[2][0], p[1][0], p[0][0]},
			{p[2][1], p[1][1], p[0][1]},
			{p[2][2], p[1][2], p[0][2]}}
	}
	pat_vertflip := func(p [][]byte) [][]byte {
		return [][]byte{p[2], p[1], p[0]}
	}
	pat_horizflip := func(p [][]byte) [][]byte {
		return [][]byte{{p[0][2], p[0][1], p[0][0]},
			{p[1][2], p[1][1], p[1][0]},
			{p[2][2], p[2][1], p[2][0]}}
	}
	pat_swapcolors := func(p [][]byte) [][]byte {
		l := [][]byte{}
		for _, s := range p {
			s = bytes.Replace(s, []byte{'X'}, []byte{'Z'}, -1)
			s = bytes.Replace(s, []byte{'x'}, []byte{'z'}, -1)
			s = bytes.Replace(s, []byte{'O'}, []byte{'X'}, -1)
			s = bytes.Replace(s, []byte{'o'}, []byte{'x'}, -1)
			s = bytes.Replace(s, []byte{'Z'}, []byte{'O'}, -1)
			s = bytes.Replace(s, []byte{'z'}, []byte{'o'}, -1)
			l = append(l, s)
		}
		return l
	}
	var pat_wildexp func(p []byte, c byte, to []byte) [][]byte
	pat_wildexp = func(p []byte, c byte, to []byte) [][]byte {
		i := bytes.Index(p, []byte{c})
		if i == -1 {
			return append([][]byte{}, p)
		}
		l := [][]byte{}
		for _, t := range bytes.Split(to, []byte{}) {
			l = append(l, pat_wildexp(append(append(append([]byte{}, p[:i]...), t[0]), p[i+1:]...), c, to)...)
		}
		return l
	}
	pat_wildcards := func(pat []byte) [][]byte {
		l := [][]byte{}
		for _, p1 := range pat_wildexp(pat, '?', []byte{'.', 'X', 'O', ' '}) {
			for _, p2 := range pat_wildexp(p1, 'x', []byte{'.', 'O', ' '}) {
				for _, p3 := range pat_wildexp(p2, 'o', []byte{'.', 'X', ' '}) {
					l = append(l, p3)
				}
			}
		}
		return l
	}

	rl := [][]byte{}
	for _, p1 := range [][][]byte{pat, pat_rot90(pat)} {
		for _, p2 := range [][][]byte{p1, pat_vertflip(p1)} {
			for _, p3 := range [][][]byte{p2, pat_horizflip(p2)} {
				for _, p4 := range [][][]byte{p3, pat_swapcolors(p3)} {
					for _, p5 := range pat_wildcards(bytes.Join(p4, []byte{})) {
						rl = append(rl, p5)
					}
				}
			}
		}
	}
	return rl
}

func pat3set_func() map[string]struct{} {
	m := make(map[string]struct{})
	for _, p := range pattern3x3Source {
		for _, s := range pat3_expand(p) {
			s = bytes.Replace(s, []byte{'O'}, []byte{'x'}, -1)
			m[string(s)] = struct{}{}
		}
	}
	return m
}

var pat3set map[string]struct{}

var patternGridcularSequence = [][][]int{ // Sequence of coordinate offsets of progressively wider diameters in gridcular metric
	{{0, 0},
		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
		{1, 1}, {-1, 1}, {1, -1}, {-1, -1}}, // d=1,2 is not considered separately
	{{0, 2}, {0, -2}, {2, 0}, {-2, 0}},
	{{1, 2}, {-1, 2}, {1, -2}, {-1, -2}, {2, 1}, {-2, 1}, {2, -1}, {-2, -1}},
	{{0, 3}, {0, -3}, {2, 2}, {-2, 2}, {2, -2}, {-2, -2}, {3, 0}, {-3, 0}},
	{{1, 3}, {-1, 3}, {1, -3}, {-1, -3}, {3, 1}, {-3, 1}, {3, -1}, {-3, -1}},
	{{0, 4}, {0, -4}, {2, 3}, {-2, 3}, {2, -3}, {-2, -3}, {3, 2}, {-3, 2}, {3, -2}, {-3, -2}, {4, 0}, {-4, 0}},
	{{1, 4}, {-1, 4}, {1, -4}, {-1, -4}, {3, 3}, {-3, 3}, {3, -3}, {-3, -3}, {4, 1}, {-4, 1}, {4, -1}, {-4, -1}},
	{{0, 5}, {0, -5}, {2, 4}, {-2, 4}, {2, -4}, {-2, -4}, {4, 2}, {-4, 2}, {4, -2}, {-4, -2}, {5, 0}, {-5, 0}},
	{{1, 5}, {-1, 5}, {1, -5}, {-1, -5}, {3, 4}, {-3, 4}, {3, -4}, {-3, -4}, {4, 3}, {-4, 3}, {4, -3}, {-4, -3}, {5, 1}, {-5, 1}, {5, -1}, {-5, -1}},
	{{0, 6}, {0, -6}, {2, 5}, {-2, 5}, {2, -5}, {-2, -5}, {4, 4}, {-4, 4}, {4, -4}, {-4, -4}, {5, 2}, {-5, 2}, {5, -2}, {-5, -2}, {6, 0}, {-6, 0}},
	{{1, 6}, {-1, 6}, {1, -6}, {-1, -6}, {3, 5}, {-3, 5}, {3, -5}, {-3, -5}, {5, 3}, {-5, 3}, {5, -3}, {-5, -3}, {6, 1}, {-6, 1}, {6, -1}, {-6, -1}},
	{{0, 7}, {0, -7}, {2, 6}, {-2, 6}, {2, -6}, {-2, -6}, {4, 5}, {-4, 5}, {4, -5}, {-4, -5}, {5, 4}, {-5, 4}, {5, -4}, {-5, -4}, {6, 2}, {-6, 2}, {6, -2}, {-6, -2}, {7, 0}, {-7, 0}},
}

//######################
// Initialization
//     collect all dynamic initializations into a callable function

// performInitialization() collects all the runtime initializations together
// so they can be called from main() allowing better control
func performInitialization() {
	emptyBoard = append(append(append(bytes.Repeat([]byte{' '}, N+1), '\n'),
		bytes.Repeat(append([]byte{' '}, append(bytes.Repeat([]byte{'.'}, N), '\n')...), N)...),
		bytes.Repeat([]byte{' '}, N+2)...)

	pat3set = pat3set_func()

	spatPatternDict = make(map[int]uint64)   // hash(neighborhoodGridcular()) <- spatial id
	largePatterns = make(map[uint64]float64) // hash(neighborhoodGridcular()) -> probability
}

//######################
// board string routines

// generator of coordinates for all neighbors of c
func neighbors(c int) []int {
	return []int{c - 1, c + 1, c - W, c + W}
}

// generator of coordinates for all diagonal neighbors of c
func diagonalNeighbors(c int) []int {
	return []int{c - W - 1, c - W + 1, c + W - 1, c + W + 1}
}

func boardPut(board []byte, c int, p byte) []byte {
	board[c] = p
	return board
}

// replace continuous-color area starting at c with special color #
func floodfill(board []byte, c int) []byte {
	localBoard := make([]byte, len(board))
	copy(localBoard, board)

	// XXX: Use bytearray to speed things up? (still needed in golang?)
	p := localBoard[c]
	localBoard = boardPut(localBoard, c, '#')
	fringe := []int{c}
	for len(fringe) > 0 {
		c, fringe = fringe[len(fringe)-1], fringe[:len(fringe)-1]
		for _, d := range neighbors(c) {
			if localBoard[d] == p {
				localBoard = boardPut(localBoard, d, '#')
				fringe = append(fringe, d)
			}
		}
	}
	return localBoard
}

// test if point of color p is adjecent to color # anywhere
// on the board; use in conjunction with floodfill for reachability
func contact(board []byte, p byte) int {
	for i := W; i < len(board)-W-1; i++ {
		if board[i] == '#' {
			for _, j := range neighbors(i) {
				if board[j] == p {
					return j
				}
			}
		}
	}
	return NONE
}

// Use Mersenne Twister as Random Number Generator in place of default
// Use multiple RNG sources to reduce contention
var rng *rand.Rand

func newRNG() *rand.Rand {
	rng := rand.New(mt64.New())
	rng.Seed(time.Now().UnixNano())
	return rng
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
func SwapCase(str []byte) []byte {
	b := make([]byte, len(str))
	for i, c := range str {
		r := rune(c)
		if unicode.IsUpper(r) {
			b[i] = byte(unicode.ToLower(r))
		} else {
			b[i] = byte(unicode.ToUpper(r))
		}
	}
	return b
}

// shuffle slice elements
func shuffleInt(a []int, rng *rand.Rand) {
	for i := len(a) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

func shuffleTree(a []*TreeNode, rng *rand.Rand) {
	for i := len(a) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
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

func patternInSet(strSlice map[string]struct{}, strTest []byte) bool {
	_, exists := strSlice[string(strTest)]
	return exists
}

// test for edge of board
func isSpace(b byte) bool {
	return bytes.Contains([]byte{' ', '\n'}, []byte{b})
}

// End of functions added to replace Python functions and methods

// test if c is inside a single-color diamond and return the diamond
// color or None; this could be an eye, but also a false one
func isEyeish(board []byte, c int) byte {
	var eyeColor, otherColor byte
	for _, d := range neighbors(c) {
		if isSpace(board[d]) {
			continue
		}
		if board[d] == '.' {
			return 0
		}
		if eyeColor == 0 {
			eyeColor = board[d]
			otherColor = byte(swapCase(rune(eyeColor)))
		} else {
			if board[d] == otherColor {
				return 0
			}
		}
	}
	return eyeColor
}

// test if c is an eye and return its color or None
func isEye(board []byte, c int) byte {
	eyeColor := isEyeish(board, c)
	if eyeColor == 0 {
		return 0
	}

	// Eye-like shape, but it could be a falsified eye
	falseColor := byte(swapCase(rune(eyeColor)))
	falseCount := 0
	atEdge := false
	for _, d := range diagonalNeighbors(c) {
		if isSpace(board[d]) {
			atEdge = true
		} else {
			if board[d] == falseColor {
				falseCount += 1
			}
		}
	}
	if atEdge {
		falseCount += 1
	}
	if falseCount >= 2 {
		return 0
	}
	return eyeColor
}

// Implementation of simple Chinese Go rules

// Position holds the current state of a the game including the stones on the
// board, the captured stones, the current Ko state, the last 2 moves, and the
// komi used.
type Position struct {
	board []byte // string representation of board state
	cap   []int  // holds total captured stones for each player
	n     int    // n is how many moves were played so far
	ko    int    // location of prohibited move under simple ko rules
	last  int    // previous move
	last2 int    // antepenultimate move
	komi  float64
}

// play as player X at the given coord c, return the new position
func (p Position) move(c int) (Position, string) {
	// Test for ko
	if c == p.ko {
		return p, "ko"
	}
	// Are we trying to play in enemy's eye?
	inEnemyEye := isEyeish(p.board, c) == 'x'

	board := make([]byte, len(p.board))
	copy(board, p.board)
	board = boardPut(board, c, 'X')
	// Test for captures, and track ko
	capX := p.cap[0]
	singleCaps := []int{}
	for _, d := range neighbors(c) {
		if board[d] != 'x' {
			continue
		}
		// XXX: The following is an extremely naive and SLOW approach
		// at things - to do it properly, we should maintain some per-group
		// data structures tracking liberties.
		fillBoard := floodfill(board, d) // get a board with the adjacent group replaced by '#'
		if contact(fillBoard, '.') != NONE {
			continue // some liberties left
		}
		// no liberties left for this group, remove the stones!
		captureCount := bytes.Count(fillBoard, []byte{'#'})
		if captureCount == 1 {
			singleCaps = append(singleCaps, d)
		}
		capX += captureCount
		board = bytes.Replace(fillBoard, []byte{'#'}, []byte{'.'}, -1) // capture the group
	}
	// set ko
	ko := NONE
	if inEnemyEye && len(singleCaps) == 1 {
		ko = singleCaps[0]
	}
	// Test for suicide
	if contact(floodfill(board, c), '.') == NONE {
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
func (p Position) passMove() (Position, string) {
	p.board = SwapCase(p.board)
	p.cap = []int{p.cap[1], p.cap[0]}
	p.n = p.n + 1
	p.ko = NONE
	p.last2 = p.last // must copy first
	p.last = PASS
	p.komi = p.komi

	return p, "ok"
}

// Generate a list of moves (includes false positives - suicide moves;
// does not include true-eye-filling moves), starting from a given board
// index (that can be used for randomization)
func (p Position) moves(i0 int, done chan struct{}) chan int {
	c := make(chan int)

	go func() {
		defer close(c)

		i := i0 - 1
		passes := 0
		for {
			index := bytes.Index(p.board[i+1:], []byte{'.'})
			if passes > 0 && (index == -1 || i+index+1 >= i0) {
				return // we have looked through the whole board
			}
			if index == -1 {
				i = 0
				passes += 1
				continue // go back and start from the beginning
			}
			i += index + 1
			// Test for to-play player's one-point eye
			if isEye(p.board, i) == 'X' {
				continue
			}
			select {
			case c <- i:
			case <-done:
				return
			}
		}

		// close(c) defer'd
	}()
	return c
}

// generate a randomly shuffled list of points including and
// surrounding the last two moves (but with the last move having
// priority)
func (p Position) lastMovesNeighbors(rng *rand.Rand) []int {
	cList := []int{}
	dList := []int{}
	for _, c := range []int{p.last, p.last2} {
		if c < 0 { // if there was no last move, or pass
			continue
		}
		dList = append([]int{c}, append(neighbors(c), diagonalNeighbors(c)...)...)
		shuffleInt(dList, rng)
		for _, d := range dList {
			if intInSlice(cList, d) {
				continue
			}
			cList = append(cList, d)
		}
	}
	return cList
}

// compute score for to-play player; this assumes a final position
// with all dead stones captured; if owner_map is passed, it is assumed
// to be an array of statistics with average owner at the end of the game
// (+1 black, -1 white)
func (p Position) score(owner_map []float64) float64 {
	board := make([]byte, len(p.board))
	copy(board, p.board)
	var fillBoard []byte
	var touches_X, touches_x bool
	var komi float64
	var n float64
	i := 0
	for {
		index := bytes.Index(p.board[i+1:], []byte{'.'})
		if index == -1 {
			break
		}
		i += index + 1
		fillBoard = floodfill(board, i)
		// fillBoard is board with some continuous area of empty space replaced by #
		touches_X = contact(fillBoard, 'X') != NONE
		touches_x = contact(fillBoard, 'x') != NONE
		if touches_X && !touches_x {
			board = bytes.Replace(fillBoard, []byte{'#'}, []byte{'X'}, -1)
		} else if touches_x && !touches_X {
			board = bytes.Replace(fillBoard, []byte{'#'}, []byte{'x'}, -1)
		} else {
			board = bytes.Replace(fillBoard, []byte{'#'}, []byte{':'}, -1) // seki, rare
		}
	}
	// now that area is replaced either by X, x or :
	if p.n%2 == 1 {
		komi = p.komi
	} else {
		komi = -p.komi
	}
	if len(owner_map) > 0 {
		for c := 0; c < W*W; c++ {
			if board[c] == 'X' {
				n = 1
			} else if board[c] == 'x' {
				n = -1
			} else {
				n = 0
			}
			if p.n%2 == 1 {
				n = -n
			}
			owner_map[c] += n
		}
	}
	return float64(bytes.Count(board, []byte{'X'})-bytes.Count(board, []byte{'x'})) + komi
}

// Return an initial board position
func emptyPosition() Position {
	var p Position

	p.board = emptyBoard
	p.cap = []int{0, 0}
	p.n = 0
	p.ko = NONE
	p.last = NONE
	p.last2 = NONE
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
// singlePointOK means that we will not try to save one-point groups;
// twoLibertyTest means that we will check for 2-liberty groups which are
// threatened by a ladder
// twoLibertyTestAtEdgeOnly means that we will check the 2-liberty groups only
// at the board edge, allowing check of the most common short ladders
// even in the playouts
func fixAtari(pos Position, c int, singlePointOK, twoLibertyTest, twoLibertyTestAtEdgeOnly bool) (bool, []int) {
	// check if a capturable ladder is being pulled out at c and return
	// a move that continues it in that case; expects its two liberties as
	// l1, l2  (in fact, this is a general 2-lib capture exhaustive solver)
	readLadderAttack := func(pos Position, c, l1, l2 int) int {
		for _, l := range []int{l1, l2} {
			pos_l, pos_err := pos.move(l)
			if pos_err != "ok" {
				continue
			}
			// fixAtari() will recursively call readLadderAttack() back;
			// however, ignore 2lib groups as we don't have time to chase them
			isAtari, atariEscape := fixAtari(pos_l, c, false, false, false)
			if isAtari && len(atariEscape) == 0 {
				return l
			}
		}
		return NONE
	}

	fillBoard := floodfill(pos.board, c)
	groupSize := bytes.Count(fillBoard, []byte{'#'})
	if singlePointOK && groupSize == 1 {
		return false, []int{}
	}
	// Find a liberty
	l := contact(fillBoard, '.')
	// Ok, any other liberty?
	fillBoard = boardPut(fillBoard, l, 'L')
	l2 := contact(fillBoard, '.')
	if l2 != NONE {
		// At least two liberty group...
		if twoLibertyTest && groupSize > 1 &&
			(!twoLibertyTestAtEdgeOnly || lineHeight(l) == 0 && lineHeight(l2) == 0) &&
			contact(boardPut(fillBoard, l2, 'L'), '.') == NONE {
			// Exactly two liberty group with more than one stone.  Check
			// that it cannot be caught in a working ladder; if it can,
			// that's as good as in atari, a capture threat.
			// (Almost - N/A for countercaptures.)
			ladderAttack := readLadderAttack(pos, c, l, l2)
			if ladderAttack >= 0 {
				return false, []int{ladderAttack}
			}
		}
		return false, []int{}
	}

	// In atari! If it's the opponent's group, that's enough...
	if pos.board[c] == 'x' {
		return true, []int{l}
	}

	solutions := []int{}

	// Before thinking about defense, what about counter-capturing
	// a neighboring group?
	counterCaptureBoard := fillBoard
	for {
		otherGroup := contact(counterCaptureBoard, 'x')
		if otherGroup == NONE {
			break
		}
		a, ccls := fixAtari(pos, otherGroup, false, false, false)
		if a && len(ccls) > 0 {
			solutions = append(solutions, ccls...)
		}
		// XXX: floodfill is better for big groups
		counterCaptureBoard = boardPut(counterCaptureBoard, otherGroup, '%')
	}

	// We are escaping.  Will playing our last liberty gain
	// at least two liberties?  Re-floodfill to account for connecting
	escpos, escerr := pos.move(l)
	if escerr != "ok" {
		return true, solutions // oops, suicidal move
	}
	fillBoard = floodfill(escpos.board, l)
	l_new := contact(fillBoard, '.')
	fillBoard = boardPut(fillBoard, l_new, 'L')
	l_new_2 := contact(fillBoard, '.')
	if l_new_2 != NONE {
		if len(solutions) > 0 ||
			!(contact(boardPut(fillBoard, l_new_2, 'L'), '.') == NONE &&
				readLadderAttack(escpos, l, l_new, l_new_2) != NONE) {
			solutions = append(solutions, l)
		}
	}
	return true, solutions
}

// return a board map listing common fate graph distances from
// a given point - this corresponds to the concept of locality while
// contracting groups to single points
func cfgDistance(board []byte, c int) []int {
	cfg_map := []int{}
	for i := 0; i < W*W; i++ {
		cfg_map = append(cfg_map, NONE)
	}
	cfg_map[c] = 0

	// flood-fill like mechanics
	fringe := []int{c}
	for len(fringe) > 0 {
		c, fringe = fringe[len(fringe)-1], fringe[:len(fringe)-1]
		for _, d := range neighbors(c) {
			if isSpace(board[d]) ||
				(0 <= cfg_map[d] && cfg_map[d] <= cfg_map[c]) {
				continue
			}
			cfg_before := cfg_map[d]
			if board[d] != '.' && board[d] == board[c] {
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
func lineHeight(c int) int {
	row, col := (c-(W+1))/W, (c-(W+1))%W
	minVal := row
	for _, testVal := range []int{col, N - 1 - row, N - 1 - col} {
		if testVal < minVal {
			minVal = testVal
		}
	}
	return minVal
}

// Check whether there are any stones in Manhattan distance up to dist
func emptyArea(board []byte, c, dist int) bool {
	for d := range neighbors(c) {
		if bytes.Contains([]byte{'X', 'x'}, []byte{board[d]}) {
			return false
		}
		if board[d] == '.' && dist > 1 && !emptyArea(board, d, dist-1) {
			return false
		}
	}
	return true
}

// return a string containing the 9 points forming 3x3 square around
//  certain move candidate
func neighborhood3x3(board []byte, c int) []byte {
	localBoard := append([]byte{}, board[c-W-1:c-W+2]...)
	localBoard = append(localBoard, board[c-1:c+2]...)
	localBoard = append(localBoard, board[c+W-1:c+W+2]...)
	return bytes.Replace(localBoard, []byte{'\n'}, []byte{' '}, -1)
}

// large-scale pattern routines (those patterns living in patterns.{spat,prob} files)

// are you curious how these patterns look in practice? get
// https://github.com/pasky/pachi/blob/master/tools/pattern_spatial_show.pl
// and try e.g. ./pattern_spatial_show.pl 71

var spatPatternDict map[int]uint64 // hash(neighborhoodGridcular()) <- spatial id

// load dictionary of positions, translating them to numeric ids
func loadSpatPatternDict(f *os.File) {
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		// line: 71 6 ..X.X..OO.O..........#X...... 33408f5e 188e9d3e 2166befe aa8ac9e 127e583e 1282462e 5e3d7fe 51fc9ee
		if strings.HasPrefix(line, "#") {
			continue
		}
		lineFields := strings.SplitN(line, " ", 4)
		id, err := strconv.ParseInt(string(lineFields[0]), 10, 0)
		if err != nil {
			continue
		}
		neighborhood := strings.Replace(strings.Replace(lineFields[2], "#", " ", -1), "O", "x", -1)

		spatPatternDict[int(id)] = xxhash.Checksum64([]byte(neighborhood))
	}
}

var largePatterns map[uint64]float64 // hash(neighborhoodGridcular()) -> probability

// dictionary of numeric pattern ids, translating them to probabilities
// that a move matching such move will be played when it is available
func loadLargePatterns(f *os.File) {
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		// line: 0.004 14 3842 (capture:17 border:0 s:784)
		lineFields := strings.SplitN(line, " ", 4)

		prob, err := strconv.ParseFloat(lineFields[0], 64)
		if err != nil {
			continue
		}
		targetData := strings.SplitN(lineFields[3], "s:", 2)
		if len(targetData) < 2 {
			continue
		}
		dataStrings := strings.SplitN(targetData[1], ")", 2)
		if dataStrings == nil {
			continue
		}
		id, err := strconv.ParseInt(dataStrings[0], 10, 0)
		if err != nil {
			continue
		}
		patternHash, goodHash := spatPatternDict[int(id)]
		if !goodHash {
			continue
		}
		largePatterns[patternHash] = prob
	}
}

// Yield progressively wider-diameter gridcular board neighborhood
// stone configuration strings, in all possible rotations
func neighborhoodGridcular(board []byte, c int, done chan struct{}) chan []byte {
	ch := make(chan []byte)

	go func() {
		defer close(ch)
		// Each rotations element is (xyindex, xymultiplier)
		rotations := [][][]int{
			{{0, 1}, {1, 1}},
			{{0, 1}, {-1, 1}},
			{{0, 1}, {1, -1}},
			{{0, 1}, {-1, -1}},
			{{1, 0}, {1, 1}},
			{{1, 0}, {-1, 1}},
			{{1, 0}, {1, -1}},
			{{1, 0}, {-1, -1}},
		}
		neighborhood := [][]byte{}
		for ri := 0; ri < len(rotations); ri++ {
			neighborhood = append(neighborhood, []byte{})
		}
		wboard := bytes.Replace(board, []byte{'\n'}, []byte{' '}, -1)
		for _, dseq := range patternGridcularSequence {
			for ri := 0; ri < len(rotations); ri++ {
				r := rotations[ri]
				for _, o := range dseq {
					y, x := (c-(W+1))/W, (c-(W+1))%W
					y += o[r[0][0]] * r[1][0]
					x += o[r[0][1]] * r[1][1]
					if y >= 0 && y < N && x >= 0 && x < N {
						si := (y+1)*W + x + 1
						neighborhood[ri] = append(neighborhood[ri], wboard[si])
					} else {
						neighborhood[ri] = append(neighborhood[ri], byte(' '))
					}
				}
				select {
				case ch <- neighborhood[ri]:
				case <-done:
					return
				}
			}
		}
		// close(ch) deferred on entry to go func()
	}()
	return ch
}

// return probability of large-scale pattern at coordinate c.
// Multiple progressively wider patterns may match a single coordinate,
// we consider the largest one.
func largePatternProbability(board []byte, c int) float64 {
	probability := float64(NONE)
	matchedLength := 0
	nonMatchedLength := 0
	done := make(chan struct{})
	for n := range neighborhoodGridcular(board, c, done) {
		prob, good_prob := largePatterns[xxhash.Checksum64(n)]
		if good_prob {
			probability = prob
			matchedLength = len(n)
			continue
		}
		if matchedLength < nonMatchedLength && nonMatchedLength < len(n) {
			// stop when we did not match any pattern with a certain
			// diameter - it ain't going to get any better!
			break
		}
		nonMatchedLength = len(n)
	}
	close(done)
	return probability
}

//##########################
// montecarlo playout policy

// Yield candidate next moves in the order of preference; this is one
// of the main places where heuristics dwell, try adding more!
//
// heuristicSet is the set of coordinates considered for applying heuristics;
// this is the immediate neighborhood of last two moves in the playout, but
// the whole board while prioring the tree.
type Result struct {
	intResult int
	strResult string
}

func generatePlayoutMoves(pos Position, heuristicSet []int, probs map[string]float64, expensiveOK bool, done chan struct{}) chan Result {
	ch := make(chan Result)
	var r Result

	go func() {
		defer close(ch)
		rng := newRNG()
		// Check whether any local group is in atari and fill that liberty
		if rng.Float64() <= probs["capture"] {
			alreadySuggested := []int{}
			for _, c := range heuristicSet {
				if bytes.Contains([]byte{'X', 'x'}, []byte{pos.board[c]}) {
					_, ds := fixAtari(pos, c, false, true, !(expensiveOK))
					shuffleInt(ds, rng)
					for _, d := range ds {
						if !(intInSlice(alreadySuggested, d)) {
							r.intResult = d
							r.strResult = "capture " + strconv.FormatInt(int64(c), 10)
							select {
							case ch <- r:
							case <-done:
								return
							}
							alreadySuggested = append(alreadySuggested, d)
						}
					}
				}
			}
		}

		// Try to apply a 3x3 pattern on the local neighborhood
		if rng.Float64() <= probs["pat3"] {
			alreadySuggested := []int{}
			for _, c := range heuristicSet {
				if pos.board[c] == '.' && !(intInSlice(alreadySuggested, c)) && patternInSet(pat3set, neighborhood3x3(pos.board, c)) {
					r.intResult = c
					r.strResult = "pat3"
					select {
					case ch <- r:
					case <-done:
						return
					}
					alreadySuggested = append(alreadySuggested, c)
				}
			}
		}

		// Try *all* available moves, but starting from a random point
		// (in other words, suggest a random move)
		moves_done := make(chan struct{})
		defer close(moves_done)
		x, y := rng.Intn(N-1)+1, rng.Intn(N-1)+1
		for c := range pos.moves(y*W+x, moves_done) {
			r.intResult = c
			r.strResult = "random"
			select {
			case ch <- r:
			case <-done:
				return
			}
		}
		// close(moves_done) defer'd
		// close(ch) defer'd at start of routine
	}()
	return ch
}

// Start a Monte Carlo playout from a given position,
// return score for to-play player at the starting position;
// amaf_map is board-sized scratchpad recording who played at a given
// position first
func mcplayout(pos Position, amaf_map []int, disp bool) (float64, []int, []float64) {
	var err string
	var pos2 Position
	var prob_reject float64
	if disp {
		fmt.Fprintln(os.Stderr, "** SIMULATION **")
	}
	rng := newRNG()

	start_n := pos.n
	passes := 0
	for passes < 2 && pos.n < MAX_GAME_LEN {
		if disp {
			printPosition(pos, os.Stderr, nil)
		}
		pos2.n = NONE
		// We simply try the moves our heuristics generate, in a particular
		// order, but not with 100% probability; this is on the border between
		// "rule-based playouts" and "probability distribution playouts".
		done := make(chan struct{})
		for r := range generatePlayoutMoves(pos, pos.lastMovesNeighbors(rng), PROB_HEURISTIC, false, done) {
			c := r.intResult
			kind := r.strResult
			if disp && kind != "random" {
				fmt.Fprintln(os.Stderr, "move suggestion", stringCoordinates(c), kind)
			}
			pos2, err = pos.move(c)
			if err != "ok" {
				pos2.n = NONE
				continue
			}
			// check if the suggested move did not turn out to be a self-atari
			if kind == "random" {
				prob_reject = PROB_RSAREJECT
			} else {
				prob_reject = PROB_SSAREJECT
			}
			if rng.Float64() <= prob_reject {
				_, atariEscape := fixAtari(pos2, c, true, true, true)
				if len(atariEscape) > 0 {
					if disp {
						fmt.Fprintln(os.Stderr, "rejecting self-atari move", stringCoordinates(c))
					}
					pos2.n = NONE
					continue
				}
			}
			if amaf_map[c] == 0 { // Mark the coordinate with 1 for black
				if pos.n%2 == 0 {
					amaf_map[c] = 1
				} else {
					amaf_map[c] = -1
				}
			}
			break
		}
		close(done)
		if pos2.n == NONE { // no valid moves, pass
			pos, _ = pos.passMove()
			passes += 1
			continue
		}
		passes = 0
		pos = pos2
	}
	owner_map := make([]float64, W*W)
	score := pos.score(owner_map)
	if disp {
		if pos.n%2 == 0 {
			fmt.Fprintf(os.Stderr, "** SCORE B%+.1f **\n", score)
		} else {
			fmt.Fprintf(os.Stderr, "** SCORE B%+.1f **\n", -score)
		}
	}
	if start_n%2 != pos.n%2 {
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
	pos      Position
	v        int // # of visits
	w        int // # wins
	pv       int // prior value of v
	pw       int // prior value of w
	av       int
	aw       int
	children []*TreeNode
}

func NewTreeNode(pos Position) *TreeNode {
	tn := new(TreeNode)
	tn.pos = pos
	tn.v = 0
	tn.w = 0
	tn.pv = PRIOR_EVEN
	tn.pw = PRIOR_EVEN / 2
	tn.av = 0
	tn.aw = 0
	tn.children = []*TreeNode{}
	return tn
}

// add and initialize children to a leaf node
func (tn *TreeNode) expand() {
	cfg_map := []int{}
	if tn.pos.last >= 0 { // there is actually a move
		cfg_map = append(cfg_map, cfgDistance(tn.pos.board, tn.pos.last)...)
	}
	tn.children = []*TreeNode{}
	childSet := map[int]*TreeNode{}
	// Use playout generator to generate children and initialize them
	// with some priors to bias search towards more sensible moves.
	// Note that there can be many ways to incorporate the priors in
	// next node selection (progressive bias, progressive widening, ...).
	seedSet := []int{}
	for i := N; i < (N+1)*W; i++ {
		seedSet = append(seedSet, i)
	}
	done := make(chan struct{})
	for r := range generatePlayoutMoves(tn.pos, seedSet, map[string]float64{"capture": 1, "pat3": 1}, true, done) {
		c := r.intResult
		kind := r.strResult
		pos2, err := tn.pos.move(c)
		if err != "ok" {
			continue
		}
		// generatePlayoutMoves() will generate duplicate suggestions
		// if a move is yielded by multiple heuristics
		node, ok := childSet[pos2.last]
		if !ok {
			node = NewTreeNode(pos2)
			tn.children = append(tn.children, node)
			childSet[pos2.last] = node
		}

		if strings.HasPrefix(kind, "capture") {
			// Check how big group we are capturing; coord of the group is
			// second word in the ``kind`` string
			coord, _ := strconv.ParseInt(strings.Split(kind, " ")[1], 10, 32)
			if bytes.Count(floodfill(tn.pos.board, int(coord)), []byte{'#'}) > 1 {
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
	close(done)

	// Second pass setting priors, considering each move just once now
	for _, node := range tn.children {
		c := node.pos.last

		if len(cfg_map) > 0 && cfg_map[c]-1 < len(PRIOR_CFG) {
			node.pv += PRIOR_CFG[cfg_map[c]-1]
			node.pw += PRIOR_CFG[cfg_map[c]-1]
		}

		height := lineHeight(c) // 0-indexed
		if height <= 2 && emptyArea(tn.pos.board, c, 3) {
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

		inAtari, _ := fixAtari(node.pos, c, true, true, false)
		if inAtari {
			node.pv += PRIOR_SELFATARI
			node.pw += 0 // negative prior
		}

		patternProbability := largePatternProbability(tn.pos.board, c)
		if patternProbability > 0.001 {
			patternPrior := math.Sqrt(patternProbability) // tone up
			node.pv += int(patternPrior * PRIOR_LARGEPATTERN)
			node.pw += int(patternPrior * PRIOR_LARGEPATTERN)
		}
	}

	if len(tn.children) == 0 {
		// No possible moves, add a pass move
		pass_pos, _ := tn.pos.passMove()
		tn.children = append(tn.children, NewTreeNode(pass_pos))
	}
}

func (tn *TreeNode) raveUrgency() float64 {
	v := tn.v + tn.pv
	expectation := float64(tn.w+tn.pw) / float64(v)
	if tn.av == 0 {
		return expectation
	}
	raveExpectation := float64(tn.aw) / float64(tn.av)
	beta := float64(tn.av) / (float64(tn.av+v) + float64(v)*float64(tn.av)/RAVE_EQUIV)
	return beta*raveExpectation + (1-beta)*expectation
}

func (tn *TreeNode) winrate() float64 {
	if tn.v > 0 {
		return float64(tn.w) / float64(tn.v)
	} else {
		return math.NaN()
	}
}

// best move is the most simulated one
func (tn *TreeNode) bestMove() (*TreeNode, bool) {
	var maxNode *TreeNode
	if len(tn.children) == 0 {
		return nil, false
	} else {
		max_v := -1
		for _, node := range tn.children {
			if node.v > max_v {
				max_v = node.v
				maxNode = node
			}
		}
	}
	return maxNode, true
}

// Descend through the tree to a leaf
func treeDescend(tree *TreeNode, amaf_map []int, disp bool) []*TreeNode {
	tree.v += 1
	nodes := []*TreeNode{tree}
	passes := 0
	for len(nodes[len(nodes)-1].children) > 0 && passes < 2 {
		if disp {
			printPosition(nodes[len(nodes)-1].pos, os.Stderr, nil)
		}

		// Pick the most urgent child
		children := make([]*TreeNode, len(nodes[len(nodes)-1].children))
		copy(children, nodes[len(nodes)-1].children)
		if disp {
			for _, child := range children {
				dumpSubtree(child, N_SIMS/50, 0, os.Stderr, false)
			}
		}
		shuffleTree(children, rng) // randomize the max in case of equal urgency

		// find most urgent child by node.raveUrgency()
		node := children[0]
		maxRave := node.raveUrgency()
		for i, c := range children {
			if i == 0 { // skip item 0 as we already have its data
				continue
			}
			testRave := c.raveUrgency()
			if testRave > maxRave {
				node = c
				maxRave = testRave
			}
		}
		nodes = append(nodes, node)

		if disp {
			fmt.Fprintf(os.Stderr, "chosen %s\n", stringCoordinates(node.pos.last))
		}
		if node.pos.last == PASS {
			passes += 1
		} else {
			passes = 0
			if amaf_map[node.pos.last] == 0 { // Mark the coordinate with 1 for black
				if nodes[len(nodes)-2].pos.n%2 == 0 {
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
func treeUpdate(nodes []*TreeNode, amaf_map []int, score float64, disp bool) {
	localNodes := nodes
	for i, j := 0, len(localNodes)-1; i < j; i, j = i+1, j-1 { // reverse the order
		localNodes[i], localNodes[j] = localNodes[j], localNodes[i]
	}
	for _, node := range localNodes {
		if disp {
			fmt.Fprintln(os.Stderr, "updating", stringCoordinates(node.pos.last), score < 0)
		}
		win := 0
		if score < 0 { // score is for to-play, node statistics for just-played
			win = 1
		}
		node.w += win
		// Update the node children AMAF stats with moves we made
		// with their color
		amaf_map_value := 1
		if node.pos.n%2 != 0 {
			amaf_map_value = -1
		}
		if len(node.children) > 0 {
			for _, child := range node.children {
				if child.pos.last == PASS {
					continue
				}
				if amaf_map[child.pos.last] == amaf_map_value {
					if disp {
						fmt.Fprintln(os.Stderr, "  AMAF updating", stringCoordinates(child.pos.last), score > 0)
					}
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

// Perform MCTS search from a given position for a given #iterations
func treeSearch(tree *TreeNode, n int, owner_map []float64, disp bool) *TreeNode {
	// Initialize root node
	if len(tree.children) == 0 {
		tree.expand()
	}

	// We could simply run treeDescend(), mcplayout(), treeUpdate()
	// sequentially in a loop.  This is essentially what the code below
	// does, if it seems confusing!

	// However, we also have an easy (though not optimal) way to parallelize
	// by distributing the mcplayout() calls to other processes using the
	// multiprocessing Python module.  mcplayout() consumes maybe more than
	// 90% CPU, especially on larger boards.  (Except that with large patterns,
	// expand() in the tree descent phase may be quite expensive - we can tune
	// that tradeoff by adjusting the EXPAND_VISITS constant.)

	numWorkers := runtime.NumCPU()
	if disp { // set to 1 when debugging
		numWorkers = 1
	}

	type Job struct {
		nodes     []*TreeNode
		amaf_map  []int
		owner_map []float64
		score     float64
	}
	type JobResult struct {
		n   int
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
			nodes := treeDescend(tree, amaf_map, disp)
			var job Job
			job.nodes = nodes
			job.amaf_map = amaf_map
			outgoing = append(outgoing, job)
		}

		if len(ongoing) >= numWorkers {
			// Too many playouts running? Wait a bit...
			time.Sleep(10 * time.Millisecond / time.Duration(numWorkers))
		} else {
			i += 1
			if i > 0 && i%REPORT_PERIOD == 0 {
				printTreeSummary(tree, i, os.Stderr)
			}

			// Issue an mcplayout job to the worker pool
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
			treeUpdate(result.nodes, result.amaf_map, result.score, disp)
			for c := 0; c < W*W; c++ {
				owner_map[c] += result.owner_map[c]
			}
		}

		// Any playouts are finished yet?
		select {
		case jobresult := <-jr: // Yes! Queue them up for storing in the tree.
			incoming = append(incoming, jobresult.job)
			delete(ongoing, jobresult.n)
		default:
		}

		// Early stop test
		best_move, ok := tree.bestMove()
		if ok {
			best_wr := best_move.winrate()
			if (i > n/20 && best_wr > FASTPLAY5_THRES) || (i > n/5 && best_wr > FASTPLAY20_THRES) {
				break
			}
		}
	}
	for len(ongoing) > 0 { // drain any pending background jobs
		jobresult := <-jr
		delete(ongoing, jobresult.n)
	}
	close(jr)

	for c := 0; c < W*W; c++ {
		owner_map[c] = owner_map[c] / float64(i)
	}
	dumpSubtree(tree, N_SIMS/50, 0, os.Stderr, true)
	printTreeSummary(tree, i, os.Stderr)
	best_move, _ := tree.bestMove()
	return best_move
}

//##################
// user interface(s)

// utility routines

// print visualization of the given board position, optionally also
// including an owner map statistic (probability of that area of board
// eventually becoming black/white)
func printPosition(pos Position, f *os.File, owner_map []float64) {
	var Xcap, Ocap int
	var board []byte
	if pos.n%2 == 0 { // to-play is black
		board = bytes.Replace(pos.board, []byte{'x'}, []byte{'O'}, -1)
		Xcap, Ocap = pos.cap[0], pos.cap[1]
	} else { // to-play is white
		board = bytes.Replace(bytes.Replace(pos.board, []byte{'X'}, []byte{'O'}, -1), []byte{'x'}, []byte{'X'}, -1)
		Ocap, Xcap = pos.cap[0], pos.cap[1]
	}
	fmt.Fprintf(f, "Move: %-3d   Black: %d caps   White: %d caps   Komi: %.1f\n", pos.n, Xcap, Ocap, pos.komi)
	prettyBoard := strings.Join(strings.Split(string(board[:]), ""), " ")
	if pos.last >= 0 {
		prettyBoard = prettyBoard[:pos.last*2-1] + "(" + string(board[pos.last:pos.last+1]) + ")" + prettyBoard[pos.last*2+2:]
	}
	pb := []string{}
	for i, row := range strings.Split(prettyBoard, "\n")[1 : N+1] {
		row = fmt.Sprintf(" %-02d%s", N-i, row[2:])
		pb = append(pb, row)
	}
	prettyBoard = strings.Join(pb, "\n")
	if len(owner_map) > 0 {
		pretty_ownermap := ""
		for c := 0; c < W*W-1; c++ {
			if isSpace(board[c]) {
				pretty_ownermap += string(board[c : c+1])
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
		pretty_ownermap = strings.Join(strings.Split(pretty_ownermap, ""), " ")
		pb2 := []string{}
		for i, orow := range strings.Split(pretty_ownermap, "\n")[1 : N+1] {
			row := fmt.Sprintf("%s  %s", pb[i], orow[1:])
			pb2 = append(pb2, row)
		}
		prettyBoard = strings.Join(pb2, "\n")
	}
	fmt.Fprintln(f, prettyBoard)
	fmt.Fprintln(f, "    "+strings.Join(strings.Split(columnString[:N], ""), " "))
	fmt.Fprintln(f, "")
}

// Sort a slice of TreeNode by the v field starting with Max v
// Return sorted slice
// This replaces a sort using a lambda function in the original Python
func bestNodes(nodes []*TreeNode) []*TreeNode {
	var i_max, v_max int

	if len(nodes) == 1 {
		return nodes
	}
	i_max = -1
	v_max = -1
	for i, node := range nodes {
		if node.v > v_max {
			i_max = i
			v_max = node.v
		}
	}
	best_nodes := []*TreeNode{nodes[i_max]}
	remaining_nodes := []*TreeNode{}
	for i := 0; i < i_max; i++ {
		remaining_nodes = append(remaining_nodes, nodes[i])
	}
	for i := i_max + 1; i < len(nodes); i++ {
		remaining_nodes = append(remaining_nodes, nodes[i])
	}
	return append(best_nodes, bestNodes(remaining_nodes)...)
}

// print this node and all its children with v >= thres.
func dumpSubtree(node *TreeNode, thres, indent int, f *os.File, recurse bool) {
	var floatValue float64
	if node.av > 0 {
		floatValue = float64(node.aw) / float64(node.av)
	} else {
		floatValue = math.NaN()
	}
	fmt.Fprintf(f, "%s+- %s %.3f (%d/%d, prior %d/%d, rave %d/%d=%.3f, urgency %.3f)\n",
		strings.Repeat(" ", indent), stringCoordinates(node.pos.last), node.winrate(),
		node.w, node.v, node.pw, node.pv, node.aw, node.av, floatValue,
		node.raveUrgency())
	if !recurse {
		return
	}
	children := node.children
	for _, child := range bestNodes(children) {
		if child.v >= thres {
			dumpSubtree(child, thres, indent+3, f, true)
		}
	}
}

func printTreeSummary(tree *TreeNode, sims int, f *os.File) {
	var exists bool
	var best_nodes []*TreeNode
	if len(tree.children) < 5 {
		best_nodes = bestNodes(tree.children)
	} else {
		best_nodes = bestNodes(tree.children)[:5]
	}
	bestSequence := []int{}
	node := tree
	for {
		bestSequence = append(bestSequence, node.pos.last)
		node, exists = node.bestMove()
		if !exists { // no children of current node
			break
		}
	}
	sequenceString := ""
	if len(bestSequence) < 6 {
		for i, c := range bestSequence {
			if i == 0 {
				continue
			}
			sequenceString += stringCoordinates(c) + " "
		}
	} else {
		for _, c := range bestSequence[1:6] {
			sequenceString += stringCoordinates(c) + " "
		}
	}
	best_nodes_string := ""
	for _, n := range best_nodes {
		best_nodes_string += fmt.Sprintf("%s(%.3f) ", stringCoordinates(n.pos.last), n.winrate())
	}
	if len(best_nodes) > 0 {
		fmt.Fprintf(f, "[%4d] winrate %.3f | seq %s | can %s\n", sims, best_nodes[0].winrate(),
			sequenceString, best_nodes_string)
	}
}

func parseCoordinates(s string) int {
	if s == "pass" {
		return PASS
	}
	row, _ := strconv.ParseInt(s[1:], 10, 32)
	col := 1 + strings.Index(columnString, strings.ToUpper(s[0:1]))
	c := W + (N-int(row))*W + col
	return c
}

func stringCoordinates(c int) string {
	if c == PASS {
		return "pass"
	}
	if c == NONE {
		return "NONE"
	}
	row, col := (c-(W+1))/W, (c-(W+1))%W
	return fmt.Sprintf("%c%d", columnString[col], N-row)
}

// various main programs

// run n Monte-Carlo playouts from empty position, return avg. score
func mcbenchmark(n int) float64 {
	var scoreSum float64
	for i := 0; i < n; i++ {
		score, _, _ := mcplayout(emptyPosition(), make([]int, W*W), false)
		scoreSum += score
	}
	return scoreSum / float64(n)
}

// A simple minimalistic text mode UI.
func gameIO(computerBlack bool) {
	reader := bufio.NewReader(os.Stdin)
	tree := NewTreeNode(emptyPosition())
	tree.expand()
	owner_map := make([]float64, W*W)
	for {
		if !(tree.pos.n == 0 && computerBlack) {
			printPosition(tree.pos, os.Stdout, owner_map)

			fmt.Print("Your move: ")
			sc, _ := reader.ReadString('\n')
			sc = strings.TrimRight(sc, " \n")
			c := parseCoordinates(sc)
			if c >= 0 {
				// Not a pass
				if tree.pos.board[c] != '.' {
					fmt.Println("Bad move (not empty point)")
					continue
				}

				// Find the next node in the game tree and proceed there
				nodes := []*TreeNode{}
				for _, node := range tree.children {
					if node.pos.last == c {
						nodes = append(nodes, node)
					}
				}
				if len(nodes) == 0 {
					fmt.Println("Bad move (rule violation)")
					continue
				}
				tree = nodes[0]
			} else {
				// Pass move
				if len(tree.children) > 0 && tree.children[0].pos.last == PASS {
					tree = tree.children[0]
				} else {
					pos, _ := tree.pos.passMove()
					tree = NewTreeNode(pos)
				}
			}
			printPosition(tree.pos, os.Stdout, nil)
		}

		owner_map = make([]float64, W*W)
		tree = treeSearch(tree, N_SIMS, owner_map, false)
		if tree.pos.last == PASS && tree.pos.last2 == PASS {
			score := tree.pos.score(owner_map)
			if tree.pos.n%2 == 1 {
				score = -score
			}
			fmt.Printf("Game over, score: B%+.1f\n", score)
			break
		}
		if float64(tree.w)/float64(tree.v) < RESIGN_THRES {
			fmt.Println("I resign.")
			break
		}
	}
	fmt.Println("Thank you for the game!")
}

// GTP interface for our program.  We can play only on the board size
// which is configured (N), and we ignore color information and assume
// alternating play!
func gtpIO() {
	gtpIn := bufio.NewScanner(os.Stdin)
	knownCommands := []string{"boardsize", "clear_board", "komi", "play",
		"genmove", "final_score", "quit", "name",
		"version", "known_command", "list_commands",
		"protocol_version", "tsdebug"}

	tree := NewTreeNode(emptyPosition())
	tree.expand()

	for gtpIn.Scan() {
		line := gtpIn.Text()
		line = strings.TrimRight(line, " \n")
		if line == "" {
			continue
		}
		line = strings.ToLower(line)
		command := strings.Split(line, " ")
		gtpCommandID := ""
		matched, _ := regexp.MatchString("\\d+", command[0])
		if matched {
			gtpCommandID = command[0]
			command = command[1:]
		}
		owner_map := make([]float64, W*W)
		ret := ""
		if command[0] == "boardsize" {
			size, _ := strconv.ParseInt(command[1], 10, 0)
			if int(size) != N {
				fmt.Fprintf(os.Stderr, "Warning: Trying to set incompatible boardsize %s (!= %d)\n", command[0], N)
				ret = "None"
			}
		} else if command[0] == "clear_board" {
			tree = NewTreeNode(emptyPosition())
			tree.expand()
		} else if command[0] == "komi" {
			komi, err := strconv.ParseFloat(command[1], 64)
			if err == nil {
				tree.pos.komi = komi
			} else {
				ret = "None"
			}
		} else if command[0] == "play" {
			c := parseCoordinates(command[2])
			if c >= 0 {
				// Find the next node in the game tree and proceed there
				nodes := []*TreeNode{}
				if len(tree.children) > 0 {
					for _, node := range tree.children {
						if node.pos.last == c {
							nodes = append(nodes, node)
						}
					}
				}
				pos, err := tree.pos.move(c)
				if err == "ok" {
					tree = NewTreeNode(pos)
				} else {
					fmt.Fprintln(os.Stderr, "Error updating sent move:", err)
					ret = "None"
				}
			} else {
				// Pass move
				if len(tree.children) > 0 && tree.children[0].pos.last == PASS {
					tree = tree.children[0]
				} else {
					pos, _ := tree.pos.passMove()
					tree = NewTreeNode(pos)
				}
			}
		} else if command[0] == "genmove" {
			tree = treeSearch(tree, N_SIMS, owner_map, false)
			if tree.pos.last == PASS {
				ret = "pass"
			} else if tree.v > 0 && float64(tree.w)/float64(tree.v) < RESIGN_THRES {
				ret = "resign"
			} else {
				ret = stringCoordinates(tree.pos.last)
			}
		} else if command[0] == "final_score" {
			score := tree.pos.score([]float64{})
			if tree.pos.n%2 == 1 {
				score = -score
			}
			if score == 0 {
				ret = "0"
			} else if score > 0 {
				ret = fmt.Sprintf("B+%.1f", score)
			} else if score < 0 {
				ret = fmt.Sprintf("W+%.1f", -score)
			}
		} else if command[0] == "name" {
			ret = "michi-go"
		} else if command[0] == "version" {
			ret = "3.0.0"
		} else if command[0] == "tsdebug" {
			printPosition(treeSearch(tree, N_SIMS, owner_map, true).pos, os.Stderr, nil)
		} else if command[0] == "list_commands" {
			ret = strings.Join(knownCommands, "\n")
		} else if command[0] == "known_command" {
			ret = "false"
			for _, known := range knownCommands {
				if command[1] == known {
					ret = "true"
					break
				}
			}
		} else if command[0] == "protocol_version" {
			ret = "2"
		} else if command[0] == "quit" {
			fmt.Printf("=%s \n\n", gtpCommandID)
			break
		} else {
			fmt.Fprintln(os.Stderr, "Warning: Ignoring unknown command -", line)
			ret = "None"
		}

		printPosition(tree.pos, os.Stderr, owner_map)
		if ret != "None" {
			fmt.Printf("=%s %s\n\n", gtpCommandID, ret)
		} else {
			fmt.Printf("?%s ???\n\n", gtpCommandID)
		}
	}
}

func main() {
	performInitialization()
	patternLoadError := false
	f, err := os.Open(spatPatternDictFile)
	if err == nil {
		fmt.Fprintln(os.Stderr, "Loading pattern spatial dictionary...")
		loadSpatPatternDict(f)
	} else {
		patternLoadError = true
		fmt.Fprintln(os.Stderr, "Error opening ", spatPatternDictFile, err)
	}
	f, err = os.Open(largePatternsFile)
	if err == nil {
		fmt.Fprintln(os.Stderr, "Loading large patterns...")
		loadLargePatterns(f)
	} else {
		patternLoadError = true
		fmt.Fprintln(os.Stderr, "Error opening ", largePatternsFile, err)
	}
	if patternLoadError {
		fmt.Fprintln(os.Stderr, "Warning: Cannot load pattern files; will be much weaker, consider lowering EXPAND_VISITS 5->2")
	}
	fmt.Fprintln(os.Stderr, "Done")

	rng = newRNG()

	if len(os.Args) < 2 {
		// Default action
		gameIO(false)
	} else if os.Args[1] == "white" {
		gameIO(true)
	} else if os.Args[1] == "gtp" {
		gtpIO()
	} else if os.Args[1] == "mcdebug" {
		score, _, _ := mcplayout(emptyPosition(), make([]int, W*W), true)
		fmt.Println(score)
	} else if os.Args[1] == "mcbenchmark" {
		fmt.Println(mcbenchmark(20))
	} else if os.Args[1] == "tsbenchmark" {
		startTime := time.Now()
		printPosition(treeSearch(NewTreeNode(emptyPosition()), N_SIMS, make([]float64, W*W), false).pos, os.Stderr, nil)
		endTime := time.Now()
		fmt.Printf("Tree search with %d playouts took %s with %d threads; speed is %.3f playouts/thread/s\n",
			N_SIMS, endTime.Sub(startTime).String(), runtime.GOMAXPROCS(0),
			float64(N_SIMS)/(endTime.Sub(startTime).Seconds()*float64(runtime.GOMAXPROCS(0))))
	} else if os.Args[1] == "tsdebug" {
		printPosition(treeSearch(NewTreeNode(emptyPosition()), N_SIMS, make([]float64, W*W), true).pos, os.Stderr, nil)
	} else {
		fmt.Fprintln(os.Stderr, "Unknown action")
	}
}
