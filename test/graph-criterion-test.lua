require 'nn'
require 'graph-criterion'

local GraphCriterionTest = torch.TestSuite()
local tester = torch.Tester()

local c = nn.GraphCriterion({
  nn.ClassNLLCriterion(),
  nn.ClassNLLCriterion(),
})

function GraphCriterionTest:with_incorrect_num_outputs()
  local f = function()
    local output = {torch.rand(2), torch.rand(3), torch.rand(4)}
    local label = {1, 1, 1}
    return c:forward(output, label)
  end
  tester:assertError(f)

  local g = function()
    local output = {torch.rand(2), torch.rand(3), torch.rand(4)}
    local label = {1, 1, 1}
    return c:backward(output, label)
  end
  tester:assertError(g)
end

function GraphCriterionTest:with_1_label()
  local output = {torch.Tensor({-0.7, -0.2}), torch.Tensor({-0.6, -0.4})}
  local label = 1
  local loss, split_losses = c:forward(output, label)
  tester:assertalmosteq(loss, 1.3, 1e-4)
  tester:assertalmosteq(split_losses[1], 0.7, 1e-4)
  tester:assertalmosteq(split_losses[2], 0.6, 1e-4)

  local grad_output = c:backward(output, label)
  tester:assertTensorEq(grad_output[1], torch.Tensor({-1, 0}), 1e-4)
  tester:assertTensorEq(grad_output[2], torch.Tensor({-1, 0}), 1e-4)
end

function GraphCriterionTest:with_a_nil_criterion()
  local nil_c = nn.GraphCriterion({
    nn.ClassNLLCriterion(),
    false
  })

  local output = {torch.Tensor({-0.7, -0.2}), torch.Tensor({-0.6, -0.4})}
  local label = 1
  local loss, split_losses = nil_c:forward(output, label)
  tester:assertalmosteq(loss, 0.7, 1e-4)
  tester:assertalmosteq(split_losses[1], 0.7, 1e-4)
  tester:assertalmosteq(split_losses[2], 0, 1e-4)

  local grad_output = nil_c:backward(output, label)
  tester:assertTensorEq(grad_output[1], torch.Tensor({-1, 0}), 1e-4)
  tester:assertTensorEq(grad_output[2], torch.Tensor({0, 0}), 1e-4)
end

tester:add(GraphCriterionTest)
tester:run()
