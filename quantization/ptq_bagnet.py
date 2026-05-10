import torch
import torch.nn as nn

def get_fusable_modules(model):

    modules_to_fuse = []

    for module_name, module in model.named_modules():

        children = list(module.named_children())
        child_names = [name for name, _ in children]
        child_modules = [m for _, m in children]

        i = 0

        while i < len(child_modules) - 1:

            # ---------------------------------------------------
            # Conv + BN + ReLU
            # ---------------------------------------------------
            if (
                    i + 2 < len(child_modules)
                    and isinstance(child_modules[i], nn.Conv2d)
                    and isinstance(child_modules[i + 1], nn.BatchNorm2d)
                    and isinstance(child_modules[i + 2], nn.ReLU)
            ):

                fuse_group = [
                    f"{module_name}.{child_names[i]}" if module_name else child_names[i],
                    f"{module_name}.{child_names[i+1]}" if module_name else child_names[i+1],
                    f"{module_name}.{child_names[i+2]}" if module_name else child_names[i+2],
                ]

                modules_to_fuse.append(fuse_group)

                i += 3
                continue

            # ---------------------------------------------------
            # Conv + BN
            # ---------------------------------------------------
            elif (
                    i + 1 < len(child_modules)
                    and isinstance(child_modules[i], nn.Conv2d)
                    and isinstance(child_modules[i + 1], nn.BatchNorm2d)
            ):

                fuse_group = [
                    f"{module_name}.{child_names[i]}" if module_name else child_names[i],
                    f"{module_name}.{child_names[i+1]}" if module_name else child_names[i+1],
                ]

                modules_to_fuse.append(fuse_group)

                i += 2
                continue

            i += 1

    return modules_to_fuse