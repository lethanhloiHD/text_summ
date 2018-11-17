import rouge

hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , " \
               "saying the meeting would not bring any progress in deadlocked negotiations to form a government ." \
               "\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting " \
               "after a series of post-election negotiations between the two opposition groups and Hun Sen 's party " \
               "to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , " \
               "but the opposition _ claiming widespread intimidation and fraud _" \
               " has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"

hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected " \
               "of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .'" \
               "\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with " \
               "foreign enemies of China '' to incite the subversion of state power , '' according to court documents " \
               "given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents " \
               "will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"

references_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected " \
               "of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .'" \
               "\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with "

references_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , " \
               "saying the meeting would not bring any progress in deadlocked negotiations to form a government ." \
               "\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting " \
               "after a series of post-election negotiations between the two opposition groups and Hun Sen 's party " \
               "to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , " \
               "but the opposition _ claiming widespread intimidation and fraud _" \
               " has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                             max_n=2,
                             limit_length=True,
                             length_limit=100,
                             length_limit_type='words',
                             # alpha=0.5, # Default F1_score
                             # weight_factor=1.2,
                             stemming=True)

scores = evaluator.get_scores(hypothesis_2, references_2)
print(scores)