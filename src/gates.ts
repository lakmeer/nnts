
import { log, rand, exp, floor, log10, limit, table, red, green } from "./utils.ts";



//
// Modelling logic gates
//

log.quiet("Loaded model: Gates");

const TRAINING_REPORT = true;


//
// Training Data Sets
//

type Weights = Array<float>;
type TrainingSet = Array<Weights>

const set_or:TrainingSet = [
  [ 0, 0, 0 ],
  [ 0, 1, 1 ],
  [ 1, 0, 1 ],
  [ 1, 1, 1 ],
]

const set_and:TrainingSet = [
  [ 0, 0, 0 ],
  [ 0, 1, 0 ],
  [ 1, 0, 0 ],
  [ 1, 1, 1 ],
]

const set_nand:TrainingSet = [
  [ 0, 0, 1 ],
  [ 0, 1, 1 ],
  [ 1, 0, 1 ],
  [ 1, 1, 0 ],
]

const set_nor:TrainingSet = [
  [ 0, 0, 1 ],
  [ 0, 1, 0 ],
  [ 1, 0, 0 ],
  [ 1, 1, 0 ],
]

const set_xor:TrainingSet = [
  [ 0, 0, 1 ],
  [ 0, 1, 0 ],
  [ 1, 0, 0 ],
  [ 1, 1, 1 ],
]


//
// Model Types
//
//
//
// Each junction is modelled explicitly with it's own params

type ModelTemplate = {
  size: number,
  forward: (...inputs:Array<float>) => float,
}

type Model = {
  name: string;
  params: Array<float>;
  forward: (...inputs:Array<float>) => float;
}

type FwdFn = (...inputs:Array<float>) => float;


// 2-input, 1-output logic gates
//  
//  x1
//    \
//     OUT(w1, w2, b) -> y
//    /
//  x2

const MODEL_GATE:ModelTemplate = {
  size: 3,
  forward: (x1, x2, ...params) => {
    const [ w1, w2, b ] = params;
    return sigmoid(x1 * w1 + x2 * w2 + b);
  }
}



// Xor is assuming 3-gate model:
//
//  x1---OR( w1,  w2, b)
//    \ /               \
//     X                 AND(w1, w2, b) -> y
//    / \               /
//  x2---NAND(w1, w2, b)

const MODEL_XOR:ModelTemplate = {
  size: 9,
  forward: (x1, x2, ...params) => {
    const fwd = MODEL_GATE.forward;
    const [ or_w1, or_w2, or_b, and_w1, and_w2, and_b, nand_w1, nand_w2, nand_b ] = params;
    return fwd(
      fwd(x1, x2, or_w1, or_w2, or_b),
      fwd(x1, x2, nand_w1, nand_w2, nand_b),
      and_w1, and_w2, and_b
    );
  }
}


const new_model = (name:string, template:ModelTemplate) => {
  return {
    name,
    forward: template.forward,
    params: Array(template.size).fill(0)
  }
}

const init_model = (name:string, template:ModelTemplate) => {
  const m = new_model(name, template);
  m.params = m.params.map(() => rand());
  return m;
}


//
// Functions
//

type CostFn = ReturnType<typeof cost_of>;

const cost_of = (model:Model, set:TrainingSet) => (...params:Weights):float => {
  let result = 0;

  for (let i = 0; i < set.length; i++) {
    const x1:float = set[i][0];
    const x2:float = set[i][1];
    const y:float = model.forward(x1, x2, ...params);
    const d = y - set[i][2];
    result += d*d;
  }

  return result / set.length;
}

const sigmoid = (x:float):float => {
  return 1 / (1 + exp(-x));
}

const confirm = (model:Model, set:TrainingSet):boolean => {
  let pass = true;
  const rows = [];

  for (let ix in set) {
    const [ x1, x2, exp ] = set[ix];
    const act = model.forward(x1, x2, ...model.params);
    const ok = exp.toString() == act.toFixed(0);
    pass = pass && ok;
    rows.push([ ok ? green("OK") : red("XX"), x1, x2, exp, act.toFixed(0) ]);
  }

  if (!pass) log.red("⚠️  Failed to converge");
  console.log(table([ 'OK', 'x1', 'x2', 'exp', 'act' ], rows));
  return pass;
}

const finite_diff = (params:Array<float>, eps:float, cost:CostFn):Array<float> => {
  const size = params.length;
  const g = Array(size);
  let c = cost(...params);
  let saved;

  for (let p = 0; p < size; p++) {
    saved = params[p];
    params[p] += eps;
    g[p] = (cost(...params) - c) / eps;
    params[p] = saved;
  }

  return g;
}


const train = (model:Model, set:TrainingSet, eps:number, rate:number, steps:number) => {

  // Model init
  let cost = cost_of(model, set);

  // Training loop
  const start = performance.now();

  for (let i = 0; i < steps; i++) {
    const grad = finite_diff(model.params, eps, cost);

    for (let p = 0; p < model.params.length; p++) {
      model.params[p] -= rate * grad[p];
    }
  }

  const time = performance.now() - start;

  // Report
  if (!TRAINING_REPORT) return;

  const c = cost(...model.params);
  const rank = costRank(c);
  log.blue(`${rank} Trained ${model.name} (${model.params.length} params) for ${steps} steps in ${time.toFixed(2)}ms`);
  log.quiet(`Final Cost:`, c);
  log.quiet(`Params: ${model.params.map(v => v.toFixed(3)).join(', ')}`);

  const passed = confirm(model, set);
}

const costRank = (n:float) => {
  const rank = -floor(log10(n));
  return [ "🔴", "🟠", "🟡", "🟢", "🔵" ][limit(0, 4, rank - 1)];
}


// Main

export const main = () => {
  const eps  = 1e-2;
  const rate = 1e-1;

  const or   = init_model("OR",   MODEL_GATE);
  const and  = init_model("AND",  MODEL_GATE);
  const nand = init_model("NAND", MODEL_GATE);
  const nor  = init_model("NOR",  MODEL_GATE);
  const xor  = init_model("XOR",  MODEL_XOR);

  train(or,   set_or,   eps, rate, 50000);
  train(and,  set_and,  eps, rate, 50000);
  train(nand, set_nand, eps, rate, 50000);
  train(nor,  set_nor,  eps, rate, 50000);
  train(xor,  set_xor,  eps, rate, 50000);

}

