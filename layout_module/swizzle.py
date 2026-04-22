from dataclasses import dataclass
from .layout import Layout


@dataclass(frozen=True)
class Swizzle:
    b_bits: int
    m_base: int
    s_shift: int
    element_bytes: int

    def __call__(self, elem_idx: int) -> int:
        byte = elem_idx * self.element_bytes
        if self.b_bits == 0:
            return elem_idx
        mask = ((1 << self.b_bits) - 1) << (self.m_base + self.s_shift)
        xor = (byte & mask) >> self.s_shift
        return (byte ^ xor) // self.element_bytes


@dataclass(frozen=True)
class ComposedLayout:
    layout: Layout
    swizzle: Swizzle

    def __call__(self, coord) -> int:
        return self.swizzle(self.layout(coord))

    def get_mode(self, i: int) -> "ComposedLayout":
        return ComposedLayout(self.layout.get_mode(i), self.swizzle)

    @property
    def shape(self):
        return self.layout.shape

    @property
    def stride(self):
        return self.layout.stride

    @property
    def rank(self) -> int:
        return self.layout.rank

    @property
    def length(self) -> int:
        return self.layout.length

    @property
    def depth(self) -> int:
        return self.layout.depth

    @property
    def size(self) -> int:
        return self.layout.size

    @property
    def cosize(self) -> int:
        return max(self(i) for i in range(self.size)) + 1


def make_composed(layout: Layout, swizzle: Swizzle) -> ComposedLayout:
    return ComposedLayout(layout, swizzle)