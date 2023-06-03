

// Strings, Formatting

import C, { colorNames } from "chalk";

export const logHelper = (name: string, color: string) => ({
  ok:    (...args) => console.log(C.green(`[${name}]`), ...args),
  info:  (...args) => console.log(C.blue(`[${name}]`), ...args),
  err:   (...args) => console.log(C.red(`[${name}]`), ...args),
  quiet: (...args) => console.log(C.grey(`[${name}]`), ...args),
})


// Math

export const rand = ():float => Math.random();


// Globalify

window.log = logHelper("NN");
window.rand = rand;

