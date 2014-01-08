open List


type network = float list list list

let sum xs = fold_left (+.) 0. xs 

let rec map3 f l1 l2 l3 = match l1, l2, l3 with 
  | [], _, _ | _, [], _  | _, _, [] -> []
  | x::xs, y::ys, z::zs -> f x y z::map3 f xs ys zs

let scan f init xs = fold_left (fun (r::rs) x -> f r x :: r :: rs) [init] xs |> rev

let rec butlast lst = match lst with
  | [] -> raise (Failure "butlast")
  | [_] -> []
  | x::xs -> x::(butlast xs)

let rec last lst = match lst with
  | [] -> raise (Failure "last")
  | x::[] -> x
  | x::xs -> last xs

let sigma  x = tanh x
let psigma x = 1. -. ((sigma x) ** 2.)

let rec dot xs ys = 
  match xs, ys with 
  | [], [] -> 0.
  | x::xs, y::ys -> x *. y +. dot xs ys
  | _, _ -> raise (Failure "dot")

let rec transpose matrix = match matrix with
  | []::_ -> []
  | _     -> map hd matrix :: transpose (map tl matrix)

let rec rlist l = match l with
  | 0 -> []
  | _ -> (Random.float 1.) :: rlist (l - 1)

(* return a list of colums of randoms 0-1 *)
let rec random_matrix rows cols = match cols with
  | 0 -> []
  | _ -> rlist rows :: random_matrix rows (cols - 1)


(* NN-specific code *)

let rec initialize_network_weights layers : network = match layers with
  | n::nn::rst -> random_matrix (n + 1) nn :: initialize_network_weights (nn::rst)
  | _ -> []

(* Runs the input through the network, return a list of (inputs * activations) *)
let feed_fwd input network =
  let b_in = 1. :: input in
  let rec runner input network = match network with      
    | [] -> []
    | weights::[] -> let node_vals     = map (dot input) weights in
                     let activated_nvs = map sigma node_vals     in
                     (node_vals, activated_nvs) :: []
    (* add bias to hidden output *)
    | weights::nw -> let node_vals     = map (dot input) weights           in
                     let activated_nvs = 1. :: map sigma node_vals         in
                     (node_vals, activated_nvs) :: runner activated_nvs nw in
  (input, b_in) :: runner b_in network

let run input network = feed_fwd input network |> last |> snd

let prop_error ds (weights, inputs) =
  let weighted_errs = map (dot ds) (tl (transpose weights)) in
  map2 (fun i s -> (psigma i) *. s) inputs weighted_errs

let update_weights rate lws acts ds =
  let acts = snd acts in
  map2 (fun n d -> map2 (fun w a -> w +. (rate *. a *. d)) n acts) lws ds

let backpropagate rate input expected network : network =
  let runs   = feed_fwd input network in
  let oins   = runs |> last |> fst    in
  let oact   = runs |> last |> snd    in
  let rruns  = runs |> rev |> tl      in
  let rnws   = rev network            in
  let oldelt = map3 (fun i a e -> (e -. a) *. (psigma i))   (* output layer deltas *)
                    oins oact expected in 
  let rldeltas = scan prop_error oldelt   (* layers' deltas, reversed *)
                      (combine rnws (map fst rruns)) in  
  map3 (update_weights rate) rnws rruns rldeltas |> rev

let backprop_set rate inputs expecteds network : network =
  fold_left (fun nw (input, expected) -> backpropagate rate input expected nw)
            network (combine inputs expecteds)

let rec train epochs rate inputs expecteds network = match epochs with
  | 0 -> network
  | e -> let new_network = backprop_set rate inputs expecteds network in
         train (epochs - 1) rate inputs expecteds new_network


(* Error and running *)

let answer input network =
  map (fun x -> if x >= 0.5 then 1 else 0) (run input network)

let error input expected network =
  sum (map2 (fun o ex -> (0.5 *. ((ex -. o) ** 2.)))
            (run input network) expected)

let set_error inputs expecteds network =
  sum (map2 (fun is es -> error is es network) inputs expecteds)


(* Testing...  *)

(* let test_inputs  = [[1.; 1.]; [1.; 0.]; [0.; 1.]; [0.; 0.]] (\* logic gate inputs *\) *)
(* let test_outputs = [[1.; 1.; 0.; 0.]; [0.; 1.; 1.; 0.]; [0.; 1.; 1.; 0.]; [0.; 0.; 0.; 1.]] (\* AND, OR, XOR, NOR gates *\) *)

(* let n = initialize_network_weights [2;3;4] *)
(* let tnw = train 1000 0.2 test_inputs test_outputs n *)
(*                      (\* error before: *\) *)
(* let se = set_error test_inputs test_outputs n *)
(*                      (\* error after: *\) *)
(* let after_se = set_error test_inputs test_outputs tnw *)
