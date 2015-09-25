open List


type network = float list list list

let sum xs = fold_left (+.) 0. xs

let rec map3 f l1 l2 l3 = match l1, l2, l3 with
  | [], _, _ | _, [], _  | _, _, [] -> []
  | x::xs, y::ys, z::zs -> f x y z::map3 f xs ys zs

let scan f init xs =
  let applycat (r::rs) x = f r x :: r :: rs in
  fold_left applycat [init] xs |> rev

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

(* return a list of rows of randoms 0-1 *)
let rec random_matrix rows cols = match rows with
  | 0 -> []
  | _ -> rlist cols :: random_matrix (rows - 1) cols


(* NN-specific code *)

(* Returns a list of weights for each layer. A given layer has n nodes, feeding
from a layer with m nodes. Each input also has an additional bias node feeing to
the next layer. So each matrix of weights for a given layer has a n rows (one
for each node) and m + 1 columns, one for each previous layer's outputs (the + 1
is for the bias node). *)
let rec initialize_network_weights layers : network = match layers with
  | n::nn::rst -> random_matrix nn (n + 1) :: initialize_network_weights (nn::rst)
  | _ -> []

(* Runs the input through the network, return a list of
   (node_values * activated_node_values) *)
let feed_fwd input network =
  let biased_input = 1. :: input in
  let rec runner input network = match network with
    | [] -> []
    | weights::[] -> let node_vals = map (dot input) weights in
                     let activated_nvs = map sigma node_vals in
                     (node_vals, activated_nvs) :: []
    (* add bias to hidden output *)
    | weights::nw -> let node_vals = map (dot input) weights in
                     let activated_nvs = 1. :: map sigma node_vals in
                     (node_vals, activated_nvs) :: runner activated_nvs nw in
  (input, biased_input) :: runner biased_input network

(* Take the last layer's activated (snd) outputs. *)
let run input network = feed_fwd input network |> last |> snd

(* "Backpropagates", for a single given layer n, the error from layer n+1, and
   returns the scaled gradient for each node in layer n. *)
let backpropagate_error deltas (layer_weights, node_values) =
  (* We find the error for the current layer (n) node by node, by weighing the
     error from the nodes in layer n+1 by the weights of the edges in layer n which
     connect to those nodes in n+1. *)
  let weighted_errors = map (dot deltas) (tl (transpose layer_weights)) in
  (* We take the errors in these nodes, and find the derivative at a given node
  (for each node in the layer), and multiply it by the error we at that node. *)
  map2 (fun v e -> (psigma v) *. e) node_values weighted_errors

(* Updates weights for a layer (n). *)
let update_weights learning_rate layer_weights activated_node_values deltas =
  let update_node_weights node_weights delta =
    let update_weight wt act = wt +. (learning_rate *. act *. delta) in
    map2 update_weight node_weights activated_node_values in
  map2 update_node_weights layer_weights deltas

let backpropagate learning_rate input expected network : network =
  (* We partially apply the learning rate to our update function, making it a
     more aesthetic call later in the code. *)
  let update_weights = update_weights learning_rate in
  (* Here we run this particular input through the network, keeping all the
     intermediate results. `runs` is then a list of tuples representing the
     value of each neuron and the value of the sigmoid activation function
     applied to it: [(value, sigmoid(value))]. *)
  let runs = feed_fwd input network in
  (* We need to reverse everything, as we'll be "back"propagating the error from
     now on. We'll also pull off the output layers value and activated value to
     get things going below. *)
  let ((output_inputs, output_activations)::rev_runs) = rev runs in
  (* We pull those out the values and activated values of the hidden layers. *)
  let (node_values, activated_node_values) = split rev_runs in
  (* We find out how far off we are from the expected classification. These are
     our deltas for the output layer.*)
  let output_deltas = map2 (-.) expected output_activations in
  (* Now we reverse outr network in anticipation of backpropagating the error. *)
  let rev_network = rev network in
  (* Here we backpropagate the actual errors by weighting the error from the
     next layer by the previous layer's weights, in effect diving up the "blame"
     for the error of the output layer by how much the layer prior contributed
     to its error (and so on for hidden layers). *)
  let rev_deltas = scan backpropagate_error output_deltas
                        (combine rev_network node_values) in
  (* Finally, using the deltas just generated, we update the weights of the
   network, layer by layer, according to the gradient at the point we're at. *)
  map3 update_weights rev_network activated_node_values rev_deltas
  |> rev  (* ... and we unreverse the network before returning *)


(* Runs backprop iteratively over a list of inputs. *)
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

let test_inputs  = [[1.; 1.]; [1.; 0.]; [0.; 1.]; [0.; 0.]] (* logic gate inputs *)
let test_outputs = [[0.]; [1.]; [1.]; [0.]]                 (* xor outs *)

let n = initialize_network_weights [2;2;1]
let tnw = train 10000 0.1 test_inputs test_outputs n
                     (* error before: *)
let se = set_error test_inputs test_outputs n
                     (* error after: *)
let after_se = set_error test_inputs test_outputs tnw
