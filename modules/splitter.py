from modules.models import BillData


def calculate_split(bill: BillData, participants: list[str]) -> dict[str, float]:
    if not participants:
        return {}

    totals: dict[str, float] = {p: 0.0 for p in participants}

    for item in bill.items:
        payers = item.assigned_to if item.assigned_to else participants
        share = item.total_price / len(payers)
        for p in payers:
            if p in totals:
                totals[p] += share

    extra_total = sum(c.amount for c in bill.additional_charges)
    if extra_total != 0:
        item_subtotal = sum(item.total_price for item in bill.items)
        for p in participants:
            ratio = (totals[p] / item_subtotal) if item_subtotal > 0 else (1 / len(participants))
            totals[p] += extra_total * ratio

    diff = bill.total - sum(totals.values())
    if abs(diff) < 1.0 and participants:
        totals[participants[0]] += diff

    return totals
