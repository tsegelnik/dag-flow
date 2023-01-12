# Update to Daya Bay data preservation

## Common tasks

- [x] Input renaming: `output -> input`, `corresponding_output -> output`
- [x] Automatic creation of outputs is **restricted**
- [x] Parentheses operator `()` as getter `[]` of inputs, but with creation
of the input, instead of `KeyError`
- [x] Implementing of flexible shift operators `>>` and `<<` or *using current*?
  - Now using curren implementation.
- [x] Implement `hooks`:
  - At an input connection
  - At a function evaluation
- [x] Two types of `Exceptions`:
  - connection and type checking (`non-critical` exception)
  - call function (`critical`)
- [x] Recursive close of a graph
- [x] Solve troubles with a connection of an input or output and closure
- [x] Implement 2 descriptors for the `Output`:
  - `Shape` and `dtype`
  - `Allocation` and `view`
- [x] Move `handlers` to the `binding` stage
- [x] Memory allocation:
  - See `core/transformation/TransformationEntry.cc` method `updateTypes()`
- [x] Datatype: `allocatable`, `non-alloc`
- [x] Datadescr: `dtype`, `shape`
- [x] Dict as `kwargs`:
  - `ws = WeightedSum()"`;
  -`{'weight' : data} >> ws` is the same as `data >> ws('weight')`
- [x] Logging
- [x] Inputs problem: there is a difference between node and output inputs
- [x] Update naming for the second order `input` and `output`: `parent`, `child`
- [x] `iinput` is a meta data, do not use in allocation and closure;
use `Node` to do this stuff; do not use second order `input` and `output`
- [x] Loops scheme:
  1) Close:
      - Typing:
        - Update types
        - Update shapes
      - Allocation
  2) Graph:
      - Node:
        - Inputs
        - Outputs
  3) See <https://hackmd.io/mMNrlOp7Q7i9wkVFvP4W4Q>
- [x] `Tainted`
- [x] Fix decorators
- [x] Move common checks in `typefunc` into standalone module
- [ ] Update wrapping

## Transformations

- [x] Implementing of some simple transformations with only `args` in function:
`Sum`, `Product`, `Division`, ...
- [x] Implementing of some simple transformations with `args` and `kwargs`:
`WeightedSum` with `weight`, ...
- [x] Check the style of the implementation
- [x] Update the inputs checks before evaluation
- [x] Concatenation
- [x] Update `WeightedSum`
- [ ] Implement `Integrator`

## Tests

- [x] Test the graph workflow with transformations
- [x] Test of open and closure of the several graphs

## Questions and suggestions

- [x] Should we use only `numpy.ndarray` or also `numpy.number` for single element:
  1) only `numpy.ndarray`!
- [] Should we implement `zero`, `unity` objects with automatic dimension?
