

// Strings, Formatting

import C, { colorNames } from "chalk";

export const logHelper = (name: string, color: string) => {
  const label = `[${name}]`;
  return {
    ok:    (...args) => console.log(C.green(label), ...args),
    info:  (...args) => console.log(C.blue(label), ...args),
    err:   (...args) => console.log(C.red(label), ...args),
    quiet: (...args) => console.log(C.grey(label, ...args)),
    red:   (...args) => console.log(C.red(label, ...args)),
    green: (...args) => console.log(C.green(label, ...args)),
    blue:  (...args) => console.log(C.blue(label, ...args)),
  }
}


// Math

export const rand = ():float => Math.random();


// Globalify

window.log = logHelper("NN");
window.rand = rand;

