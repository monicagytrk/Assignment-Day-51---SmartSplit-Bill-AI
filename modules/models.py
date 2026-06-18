from dataclasses import dataclass, field


@dataclass
class BillItem:
    name: str
    quantity: float
    price_per_item: float
    total_price: float
    assigned_to: list[str] = field(default_factory=list)


@dataclass
class AdditionalCharge:
    name: str
    amount: float


@dataclass
class BillData:
    items: list[BillItem] = field(default_factory=list)
    subtotal: float = 0.0
    additional_charges: list[AdditionalCharge] = field(default_factory=list)
    total: float = 0.0

    def recalculate_total(self) -> float:
        subtotal = sum(item.total_price for item in self.items)
        extras = sum(c.amount for c in self.additional_charges)
        return subtotal + extras
