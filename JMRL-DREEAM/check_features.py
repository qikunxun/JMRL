import pickle
import os
import joblib
teacher_sig_path = 'best_teacher_model'
basename = 'train_distant'
attns_file = os.path.join(teacher_sig_path, f"{basename}.attns.part1")
attns_part1 = pickle.load(open(attns_file, 'rb'))
print(len(attns_part1))
attns_file = os.path.join(teacher_sig_path, f"{basename}.attns.part2")
attns_part2 = pickle.load(open(attns_file, 'rb'))
print(len(attns_part2))
attns_part = attns_part1
attns_part += attns_part2
attns_part2 = None

attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
attns_part1 = joblib.dump(attns_part, open(attns_file, 'wb'))