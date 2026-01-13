class ModelActivations:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.handles = []

    def _save_activation(self, name):
            def hook(module, inp, out):
                # store a detached tensor to avoid keeping computation graph
                try:
                    self.activations[name] = out
                except Exception:
                    # out might be a tuple (e.g., some modules); try to detach each element
                    if isinstance(out, (list, tuple)):
                        # self.activations[name] = tuple(o.detach() if hasattr(o, 'detach') else o for o in out)
                        self.activations[name] = tuple(o for o in out)

                    else:
                        self.activations[name] = out
            return hook
    
    def register_hooks(self, layer=None):
        for name, module in self.model.named_modules():
            if layer is not None:
                if name == layer:
                    if name not in self.layers:
                        self.layers.append(layer)
                    handle = module.register_forward_hook(self._save_activation(name))
                    self.handles.append(handle)
            else:
                if name in self.layers:
                    handle = module.register_forward_hook(self._save_activation(name))
                    self.handles.append(handle)
                    print(f"Registered hook for layer: {name}")

    def fetch_activations(self, layer_name):
        return self.activations.get(layer_name, None)