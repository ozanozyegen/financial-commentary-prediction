import numpy as np
""" Features
['Sales', 'Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday',
       'SchoolHoliday', 'weekday', 'month', 'weekofmonth']
"""

def create_multiclass_promo_labels(x, config):
    """
    Returns:
        y:  High promo >= 11, Med promo 9-11, Low promo <= 9, No Promo
    """
    label_history = config.get('LABEL_HISTORY')
    y = np.zeros((x.shape[0],))
    vec = (x[:,-label_history:, 4] >= 1).sum(axis=1) # Promo vec
    y[vec >= 12] = 2
    y[np.all((vec>7, vec<=11), axis=0)] = 1
    y[np.all((vec>0, vec<=7), axis=0)] = 0
    return y


label_generators = dict(
    MULTI_CLASS_PROMO = create_multiclass_promo_labels,
)