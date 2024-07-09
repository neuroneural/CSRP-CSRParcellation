def decode_names():
    # Your list of byte strings
    byte_names = [
        b'unknown', b'bankssts', b'caudalanteriorcingulate', b'caudalmiddlefrontal', b'corpuscallosum', b'cuneus', 
        b'entorhinal', b'fusiform', b'inferiorparietal', b'inferiortemporal', b'isthmuscingulate', b'lateraloccipital', 
        b'lateralorbitofrontal', b'lingual', b'medialorbitofrontal', b'middletemporal', b'parahippocampal', b'paracentral', 
        b'parsopercularis', b'parsorbitalis', b'parstriangularis', b'pericalcarine', b'postcentral', b'posteriorcingulate', 
        b'precentral', b'precuneus', b'rostralanteriorcingulate', b'rostralmiddlefrontal', b'superiorfrontal', 
        b'superiorparietal', b'superiortemporal', b'supramarginal', b'frontalpole', b'temporalpole', 
        b'transversetemporal', b'insula'
    ]

    """Decode a list of byte strings to a list of regular strings."""
    return [name.decode('utf-8') for name in byte_names]


# Decode the byte strings to regular strings
decoded_names = decode_names()
print(decoded_names)
