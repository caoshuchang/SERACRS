"""Recover positive/negative movie entities from the original ReDial forms."""

import argparse
import json
import os
import re
from collections import defaultdict


MOVIE_PATTERN = re.compile(r"@(\d+)")


def _rows(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="/root/autodl-tmp/MSCRS/raw_data/redial")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/MSCRS/rec_data/redial")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        args.output = os.path.join(args.data_dir, "conversation_preferences.json")

    raw_dialogues = {}
    for split in ("train", "valid", "test"):
        for dialogue in _rows(os.path.join(args.raw_dir, f"{split}.jsonl")):
            raw_dialogues[int(dialogue["conversationId"])] = dialogue

    # A processed recommendation row identity is conversation/message-index.
    # Multiple targets at the same index preserve movie mention order.
    targets_by_turn = defaultdict(list)
    context_movies_by_turn = {}
    item_ids = set(json.load(open(os.path.join(args.data_dir, "item_ids.json"))))
    for split in ("train", "valid", "test"):
        for row in _rows(os.path.join(args.data_dir, f"{split}_data.jsonl")):
            conversation, turn = map(int, str(row["identity"]).split("/", 1))
            targets_by_turn[(conversation, turn)].append(int(row["items"]))
            context_movies_by_turn[(conversation, turn)] = [
                int(entity) for entity in row.get("context_entities", []) if int(entity) in item_ids
            ]

    movie_mapping = defaultdict(dict)
    global_movie_votes = defaultdict(lambda: defaultdict(int))
    exact_turns = partial_turns = 0
    for (conversation, turn), targets in targets_by_turn.items():
        dialogue = raw_dialogues.get(conversation)
        if dialogue is None or turn >= len(dialogue["messages"]):
            continue
        raw_movies = MOVIE_PATTERN.findall(dialogue["messages"][turn]["text"])
        raw_movies = list(dict.fromkeys(raw_movies))
        if len(raw_movies) == len(targets):
            exact_turns += 1
            for raw_movie, target in zip(raw_movies, targets):
                global_movie_votes[raw_movie][target] += 1
        elif raw_movies and targets:
            partial_turns += 1
        for raw_movie, target in zip(raw_movies, targets):
            movie_mapping[conversation][raw_movie] = target

    # Supplement mappings by matching the ordered mentions before a processed
    # turn to the ordered context movie entities. This covers seeker mentions
    # that never became recommendation targets.
    for (conversation, turn), context_movies in context_movies_by_turn.items():
        dialogue = raw_dialogues.get(conversation)
        if dialogue is None:
            continue
        raw_context = []
        for message in dialogue["messages"][:turn]:
            raw_context.extend(MOVIE_PATTERN.findall(message["text"]))
        raw_context = list(dict.fromkeys(raw_context))
        global_context = list(dict.fromkeys(context_movies))
        if len(raw_context) == len(global_context):
            for raw_movie, global_movie in zip(raw_context, global_context):
                movie_mapping[conversation].setdefault(raw_movie, global_movie)

    # Raw ReDial movie ids are dataset-global.  Exact turn alignments therefore
    # provide a high-coverage fallback for a title not aligned in one specific
    # conversation.  Reject ties/noisy conflicts instead of guessing.
    global_movie_mapping = {}
    for raw_movie, votes in global_movie_votes.items():
        ordered = sorted(votes.items(), key=lambda pair: pair[1], reverse=True)
        best_movie, best_count = ordered[0]
        if best_count / sum(votes.values()) >= 0.9:
            global_movie_mapping[raw_movie] = best_movie

    preferences = {}
    labelled = mapped = direct_mapped = global_mapped = positive = negative = neutral = 0
    for conversation, dialogue in raw_dialogues.items():
        positive_movies = set()
        negative_movies = set()
        neutral_movies = set()
        questions = dialogue.get("initiatorQuestions", {})
        # The original release uses [] for a small number of missing forms.
        if not isinstance(questions, dict):
            questions = {}
        for raw_movie, answer in questions.items():
            labelled += 1
            global_movie = movie_mapping[conversation].get(str(raw_movie))
            if global_movie is not None:
                direct_mapped += 1
            else:
                global_movie = global_movie_mapping.get(str(raw_movie))
                if global_movie is not None:
                    global_mapped += 1
            if global_movie is None:
                continue
            mapped += 1
            liked = int(answer.get("liked", 2))
            if liked == 1:
                positive_movies.add(global_movie)
                positive += 1
            elif liked == 0:
                negative_movies.add(global_movie)
                negative += 1
            else:
                neutral_movies.add(global_movie)
                neutral += 1
        preferences[str(conversation)] = {
            "positive": sorted(positive_movies),
            "negative": sorted(negative_movies),
            "neutral": sorted(neutral_movies),
        }

    payload = {
        "preferences": preferences,
        "statistics": {
            "raw_dialogues": len(raw_dialogues),
            "exact_aligned_turns": exact_turns,
            "partial_aligned_turns": partial_turns,
            "labelled_movies": labelled,
            "mapped_labelled_movies": mapped,
            "direct_conversation_mappings": direct_mapped,
            "global_fallback_mappings": global_mapped,
            "mapping_coverage": mapped / max(1, labelled),
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        },
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(json.dumps(payload["statistics"], indent=2))


if __name__ == "__main__":
    main()
