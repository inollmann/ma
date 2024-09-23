import pickle

save_point = 2000

# with open('vocab/translations/dgs_korpus.pkl', 'rb') as f:
#     data = pickle.load(f)
# data['exclude'] = [False] * len(data['id'])

with open('vocab/translations/dgs_korpus_v2.pkl', 'rb') as f:
    data = pickle.load(f)

for idx, de, dgs, excl in zip(data['id'][save_point:],
                              data['de'][save_point:],
                              data['dgs'][save_point:],
                              data['exclude'][save_point:]):
    print(f"{idx}: {de}")
    print(dgs)
    print("Excluded:", excl)
    kb = "z"
    while kb[0] not in ["[", "y", "n", "e"]:
        kb = input("Exclude? y = Yes, n = No, [no input] = Leave Unchanged, e = Exit")
        if kb == "":
            kb = "[no input]"
        print(kb + "\n")
    if kb[0] == "y":
        data['exclude'][idx] = True
    elif kb[0] == "n":
        data['exclude'][idx] = False
    elif kb[0] == "e":
        break

pickle.dump(data, open('vocab/translations/dgs_korpus_v2.pkl', 'wb'))
