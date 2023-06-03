
//
// Modelling logic gates
//

log.quiet("Loaded model: Gates");


//
// Training Data Sets
//

type TrainingSet = Array<[float, float, float]>;

const set_or = [
  [ 0, 0, 0 ],
  [ 0, 1, 1 ],
  [ 1, 0, 1 ],
  [ 1, 1, 1 ],
]

const set_and = [
  [ 0, 0, 0 ],
  [ 0, 1, 0 ],
  [ 1, 0, 0 ],
  [ 1, 1, 1 ],
]

const set_nand = [
  [ 0, 0, 1 ],
  [ 0, 1, 1 ],
  [ 1, 0, 1 ],
  [ 1, 1, 0 ],
]

const set_xor = [
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

const MODEL = {

  // 2-input, 1-output logic gates

  GATE: {
    size: 3,
    forward: (x1, x2, ...params) => {
      const [ w1, w2, b ] = params;
      return sigmoid(x1 * w1 + x2 * w2 + b);
    }
  },

  // Xor is assuming 3-gate model:
  //
  //  x1---OR( w1,  w2, b)
  //    \ /               \
  //     X                 AND(w1, w2, b) -> y
  //    / \               /
  //  x2---NAND(w1, w2, b)

  XOR: {
    size: 9,
    forward: (x1, x2, ...params) => {
      const fwd = MODEL.GATE.forward;
      const [ or_w1, or_w2, or_b, and_w1, and_w2, and_b, nand_w1, nand_w2, nand_b ] = params;
      return fwd(
        fwd(x1, x2, or_w1, or_w2, or_b),
        fwd(x1, x2, nand_w1, nand_w2, nand_b),
        and_w1, and_w2, and_b
      );
    }
  }
}

const init_model = (name:string, template:ModelTemplate) => {
  return {
    name,
    forward: template.forward,
    params: Array(template.size).fill(0).map(() => rand())
  }
}


//
// Functions
//

const cost_of = (set, forward) => (...params):float => {
  let result = 0;

  for (let i = 0; i < set.length; i++) {
    const x1:float = set[i][0];
    const x2:float = set[i][1];
    const y:float = forward(x1, x2, ...params);
    const d = y - set[i][2];
    result += d*d;
  }

  return result / set.length;
}

const sigmoid = (x:float):float => {
  return 1 / (1 + Math.exp(-x));
}

const confirm = (title, set, forward, params):boolean => {
  const cost = cost_of(set, forward);

  let pass = true;

  log.info(`${title}: ${cost(...params)}`);
  log.info(" x1 |  x2 | exp | act");
  log.info("----+-----+-----+----");

  for (let ix in set) {
    const [ x1, x2, exp ] = set[ix];
    const act = forward(x1, x2, ...params);
    const ok = exp == act.toFixed(0);
    const print = ok ? log.ok : log.err;
    pass = pass && ok;
    print(` ${x1}  |  ${x2}  |  ${exp}  |  ${act.toFixed(0)}  (${act})`);
  }

  return pass;
}

const train = (model:Model, set:TrainingSet, eps:number, rate:number, steps:number) => {

  // Model init
  let cost = cost_of(set, model.forward);

  // Training loop
  const start = performance.now();
  for (let i = 0; i < steps; i++) {
    let c = cost(...model.params);
    for (let p = 0; p < model.params.length; p++) {
      let dp = (cost(...model.params.map((v, ix) => ix == p ? v + eps : v)) - c) / (eps);
      model.params[p] -= rate * dp;
    }
  }
  const time = performance.now() - start;

  // Report
  console.log('');
  log.blue(`Trained ${model.name} (${model.params.length} params) for ${steps} steps in ${time.toFixed(2)}ms`);
  log.info(`Final cost:`, cost(...model.params));
  log.info(`Params: ${model.params.map(v => v.toFixed(3)).join(', ')}`);
  log.info("------------");

  const passed = confirm(model.name, set, model.forward, model.params);

  if (!passed) {
    log.red("Failed");
  } else {
    log.green("Passed");
  }
}


// Main

export const main = () => {
  const eps  = 1e-2;
  const rate = 1e-1;

  const or   = init_model("OR",   MODEL.GATE);
  const and  = init_model("AND",  MODEL.GATE);
  const nand = init_model("NAND", MODEL.GATE);
  const xor  = init_model("XOR",  MODEL.XOR);

  train(or,   set_or,   eps, rate, 50000);
  train(and,  set_and,  eps, rate, 50000);
  train(nand, set_nand, eps, rate, 50000);
  train(xor,  set_xor,  eps, rate, 50000);
}

