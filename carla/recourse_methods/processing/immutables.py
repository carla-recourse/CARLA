def encoded_immutables(imtbls, input_order):
    """
    Transforms not encoded immutable feature names into encoded ones

    Returns
    -------
    list
    """
    immutables = []
    # TODO: Maybe find a more elegant way to find encoded immutable feature names
    for feature in imtbls:
        if feature in input_order:
            immutables.append(feature)
        else:
            for cat in input_order:
                if cat not in immutables:
                    if feature in cat:
                        immutables.append(cat)
                        break
    return immutables
