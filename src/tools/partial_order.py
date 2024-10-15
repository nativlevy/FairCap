

def ordinalMapping(ordinal_attr):
    return { attr: 
                { val: idx for idx, val in enumerate(vals)} 
                for attr, vals in ordinal_attr.items()
            }

