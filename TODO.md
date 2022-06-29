
# Update to Daya Bay data preservation

## Graph `<-> dagflow`

- [x] Input renaming: `output -> input`, `corresponding_output -> output`
- [x] Automatic creation of outputs is **restricted**
- [x] Parentheses operator `()` as getter `[]` of inputs, but with creation
of the input, instead of `KeyError`
- [ ] Implementing of flexible shift operators `>>` and `<<` or *using current*?
- [ ] Implement `hooks`
- [ ] Fix bug with creation of new nodes via decorator:
only works with a single instance, the graph holds only a new instance
and forget old ones!

## Transformations

- [ ] Implementing of some simple transformations with only `args` in function:
`Sum`, `Product`, `Division`, ...
- [ ] Implementing of some simple transformations with `args` and `kwargs`:
`WeightedSum` with `weight`, ...

## Tests

- [ ] Test the graph workflow with transformations

