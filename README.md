# torch-graph-criterion
A multi-criterion which maps every output of a graph module to its own criterion

## Installation
Download this module, and run `luarocks make` inside the directory.

## Usage
Assume you have the following softmax outputs from your graph module:

```lua
{out1, out2, out3}
```

Assuming you want cross-entropy loss on each of these outputs, create
the following

```lua
require 'GraphCriterion'

-- ...

local criterion = nn.GraphCriterion({
  nn.ClassNLLCriterion(),
  nn.ClassNLLCriterion(),
  nn.ClassNLLCriterion()
})
```

On `criterion:forward({out1, out2, out3}, {label1, label2, label3})`,
`out1` and `label1` will be assigned to the first `ClassNLLCriterion`,
and so on. The computed loss will be the sum of all the criteria's losses.

### Usage notes

- The number of outputs must equal the number of criteria
- The number of labels must equal the number of criteria. Alternatively, if
  all criteria share the same label, you can pass a single label (not a table).
- If you don't want one of the outputs to be assigned to a criteria, then
  pass in `false` instead of criterion in the constructor.
