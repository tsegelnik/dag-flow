# Update to Daya Bay data preservation

## Graph `<-> dagflow`

- [x] Input renaming: `output -> input`, `corresponding_output -> output`
- [x] Automatic creation of outputs is **restricted**
- [x] Parentheses operator `()` as getter `[]` of inputs, but with creation
of the input, instead of `KeyError`
- [ ] Implementing of flexible shift operators `>>` and `<<` or *using current*?
- [ ] Fix bug with creation of new nodes via decorator:
only works with a single instance, the graph holds only a new instance
and forget old ones!
- [x] Implement `hooks`:
  - At an input connection
  - At a function evaluation
- [x] Two types of `Exceptions`:
  - connection and type checking (`non-critical` exception)
  - call function (`critical`)
- [x] Recursive close of a graph
- [x] Solve troubles with a connection of an input or output and closure
- [ ] Implement 2 descriptors for the `Output`:
  - `Shape` and `dtype`
  - `Allocation` and `view`
- [ ] Move `handlers` to the `binding` stage
- [ ] Memory allocation:
  - See `core/transformation/TransformationEntry.cc` method `updateTypes()`
- [ ] Dict as `kwargs`:
  - `{'weight' : data} >> ws('weight')`
- [ ] Datatype: `allocatable`, `non-alloc`
- [ ] Datadescr: `dtype`, `shape`
- [ ] Loops:
  - Close:
    - Update types
    - Update shapes
  - Allocate
  - See <https://hackmd.io/mMNrlOp7Q7i9wkVFvP4W4Q>
- [ ] Logging

## Transformations

- [x] Implementing of some simple transformations with only `args` in function:
`Sum`, `Product`, `Division`, ...
- [x] Implementing of some simple transformations with `args` and `kwargs`:
`WeightedSum` with `weight`, ...
- [x] Check the style of the implementation

## Tests

- [x] Test the graph workflow with transformations

