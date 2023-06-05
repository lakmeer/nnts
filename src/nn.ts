
import { logHelper } from "./utils.ts";

const log = logHelper("NN");

log.quiet("Loaded NN framework");


//
// NN.ts
//
// 🔪 Cutting edge machine-learning framework
//

import * as Mat from "./matrix.ts";

const a = Mat.alloc(2, 2);
Mat.rand(a);
Mat.print(a);

const b = Mat.alloc(2, 2);
Mat.rand(b);
Mat.print(b);

const c = Mat.dot(a, b);
Mat.print(c);

Mat.sum(a, b);
Mat.print(a);

Mat.rand(b);
Mat.print(b);

