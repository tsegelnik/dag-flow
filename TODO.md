
# Update to Daya Bay data preservation

## Graph `<-> dagflow`

- [x] Input renaming: `output -> input`, `corresponding_output -> output`
- [ ] Automatic creation of outputs is *restricted*
- [ ] Parentheses operator `()` as getter `[]` of inputs, but with creation
of the input, instead of `KeyError`
- [ ] Implementing of flexible shift operators `>>` and `<<`

## Transformations

- [ ] Implementing of some simple transformations with only `args` in function:
`Sum`, `Product`, `Division`, ...
- [ ] Implementing of some simple transformations with `args` and `kwargs`:
`WeightedSum` with `weight`, ...

## Tests

- [ ] Test the graph workflow with transformations

