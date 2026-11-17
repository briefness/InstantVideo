"""采样只提供时间区间；顺序可证实、可证伪或未决，不补造亚帧时间。"""
from bisect import bisect_left


def sampled_causal_order(evidence: list[dict], policy: dict, samples: list[float]) -> dict:
    def bounds(t):
        i = bisect_left(samples, t)
        return [samples[max(0, i - 1)], samples[min(i, len(samples) - 1)]]

    relations = []

    def compare(label, left, right, strict):
        a, b = bounds(left), bounds(right)
        # 非严格依赖允许相邻动作共用交接边界；严格因果不能用同一帧证明。
        if not strict and left == right:
            status = 'passed'
        elif left < right and a[1] <= b[0]:
            status = 'passed'
        elif left > right and a[0] >= b[1]:
            status = 'failed'
        else:
            status = 'uncertain'
        relations.append({'relation': label, 'status': status,
                          'left_bounds': a, 'right_bounds': b})

    prior = {}
    order = policy.get('event_order')
    if order is not None and (not isinstance(order, list) or len(order) != len(evidence)):
        raise ValueError('事件依赖合同格式无效')
    for index, event in enumerate(evidence):
        event_id = event.get('event_id')
        if type(event_id) is not int or event_id <= 0 or event_id in prior:
            raise ValueError('时间证据事件身份无效')
        compare(f'{event_id}:cause_before_response', event['observed_effect_seconds'],
                event['observed_reaction_seconds'], True)
        if order is not None:
            row = order[index]
            if (not isinstance(row, dict) or set(row) != {'event_id', 'after_ends', 'after_effects'}
                    or type(row['event_id']) is not int or row['event_id'] != event_id):
                raise ValueError('事件依赖与实际证据身份不一致')
            for field, earlier_key, later_key in (
                ('after_ends', 'observed_end_seconds', 'observed_start_seconds'),
                ('after_effects', 'observed_effect_seconds', 'observed_effect_seconds'),
            ):
                ids = row[field]
                if (not isinstance(ids, list) or any(type(i) is not int or i not in prior for i in ids)
                        or len(ids) != len(set(ids))):
                    raise ValueError('事件依赖必须指向已有事件，不能自引用、循环或重复')
                for previous in ids:
                    compare(f'{previous}->{event_id}:{field}', prior[previous][earlier_key],
                            event[later_key], field == 'after_effects')
        elif prior:
            last = next(reversed(prior.values()))
            compare(f'{last["event_id"]}->{event_id}:start_order', last['observed_start_seconds'],
                    event['observed_start_seconds'], False)
            compare(f'{last["event_id"]}->{event_id}:effect_order', last['observed_effect_seconds'],
                    event['observed_effect_seconds'], True)
        prior[event_id] = event
    states = {r['status'] for r in relations}
    return {'status': 'failed' if 'failed' in states else 'uncertain' if 'uncertain' in states else 'passed',
            'relations': relations}
