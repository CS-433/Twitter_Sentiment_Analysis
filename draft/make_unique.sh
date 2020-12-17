tweetsDir="twitter-datasets/"
sort "${tweetsDir}train_neg.txt" | uniq > "${tweetsDir}train_neg_u.txt"
sort "${tweetsDir}train_pos.txt" | uniq > "${tweetsDir}train_pos_u.txt"
sort "${tweetsDir}train_neg_full.txt" | uniq > "${tweetsDir}train_neg_full_u.txt"
sort "${tweetsDir}train_pos_full.txt" | uniq > "${tweetsDir}train_pos_full_u.txt"

