def ordinal(i):
    if i == 0:
        return '1st'
    elif i == 1:
        return '2nd'
    elif i == 2:
        return '3rd'
    else:
        return f'{i+1}th'
