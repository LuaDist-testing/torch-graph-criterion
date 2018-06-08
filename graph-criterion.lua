require 'nn'
local _ = require 'moses'
local GraphCriterion, Parent = torch.class('nn.GraphCriterion', 'nn.Module')

function GraphCriterion:__init(criteria)
  self.criteria = criteria
end

function GraphCriterion:cuda()
  for i = 1, #self.criteria do
    if self.criteria[i] then
      self.criteria[i] = self.criteria[i]:cuda()
    end
  end
  return self
end

function GraphCriterion:double()
  for i = 1, #self.criteria do
    if self.criteria[i] then
      self.criteria[i] = self.criteria[i]:double()
    end
  end
  return self
end

function GraphCriterion:float()
  for i = 1, #self.criteria do
    if self.criteria[i] then
      self.criteria[i] = self.criteria[i]:float()
    end
  end
  return self
end

function GraphCriterion:forward(outputs, labels)
  -- Check number of outputs
  if #outputs ~= #self.criteria then
    error('Number of outputs doesn\'t match number of criteria')
  end

  local outputs = torch.Tensor(_.map(self.criteria, function(i, criterion)
    return criterion and criterion:forward(outputs[i], labels) or 0
  end))

  -- Forward on all criteria
  self.output = 0
  for i = 1, #self.criteria do
    if self.criteria[i] then
      self.output = self.output + outputs[i]
    end
  end

  return self.output, outputs
end

function GraphCriterion:backward(outputs, labels)
  -- Check number of outputs
  if #outputs ~= #self.criteria then
    error('Number of outputs doesn\'t match number of criteria')
  end

  -- Backward on all criteria
  self.gradInput = {}
  for i = 1, #outputs do
    if self.criteria[i] then
      local grad_input = self.criteria[i]:backward(outputs[i], labels)
      table.insert(self.gradInput, grad_input)
    else
      table.insert(self.gradInput, torch.Tensor(outputs[i]:size()):fill(0):typeAs(outputs[i]))
    end
  end

  return self.gradInput
end
