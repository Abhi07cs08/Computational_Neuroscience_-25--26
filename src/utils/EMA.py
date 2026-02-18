import copy
import torch


@torch.no_grad()
def build_teacher(student: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_teacher(student: torch.nn.Module, teacher: torch.nn.Module, m: float) -> None:
    # teacher = m*teacher + (1-m)*student
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)