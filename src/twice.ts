
//
// Episode 1: Machine Learning in C
//
// aka "Training our last braincell"
//
// Single-input, single-neuron model that models f(x) -> 2x
//

log.ok("Loaded model: Twice");

const train:Array<TrainingPair> = [
  [ 0, 0 ],
  [ 1, 2 ],
  [ 2, 4 ],
  [ 3, 6 ],
  [ 4, 8 ],
]

const cost = (w:float, b:float):float => {
  let result = 0;

  for (let i = 0; i < train.length; i++) {
    const x:float = train[i][0];
    const y:float = x * w + b;
    const d = y - train[i][1];
    result += d*d;
  }

  return result / train.length;
}



// Main

export const main = () => {

  let w = rand() * 10;
  let b = rand() * 5;

  const eps  = 1e-3;
  const rate = 1e-3;

  // Finite diff method
  for (let i = 0; i < 1000; i++) {
    let dw = (cost(w + eps, b) - cost(w, b)) / (eps);
    let db = (cost(w, b + eps) - cost(w, b)) / (eps);
    w -= rate * dw;
    w -= rate * db;
  }

  log.info(`cost: ${cost(w, b).toFixed(6)}, param: ${w.toFixed(6)}, bias: ${b.toFixed(6)}`);
  log.info("------------");
  log.info(w.toFixed(3), b.toFixed(3));

}


