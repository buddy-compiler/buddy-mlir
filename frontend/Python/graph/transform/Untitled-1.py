


class Rule:

  def __init__(condition, constraint):
    pass

@register()
class QuantizationMethod:

  def rewrite(self, **stuff):
    pass

  def backward(**stuff):
    pass


  def forward(self, node, context):

    state: bool
    axis: bool

    axes = []

    if state:
      if axis:
        axes.append(None)

    
    if axes:
      return Quantized(axes)
        


  