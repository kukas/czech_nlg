class DatasetPreprocessing:
    """Class that handles dataset preprocessing pipeline"""

    def __init__(self, preprocessing_num_workers):
        self.preprocessing_num_workers = preprocessing_num_workers

    def rename_columns_cs_restaurants(self, datasets):
        return datasets.rename_column("text", "target").rename_column("da", "input")

    def rename_columns_translation_dataset(
        self, datasets, source_lang="en", target_lang="cs"
    ):
        def _preprocess_function(examples):
            inputs = [ex[source_lang] for ex in examples["translation"]]
            targets = [ex[target_lang] for ex in examples["translation"]]
            return {"input": inputs, "target": targets}

        return datasets.map(
            _preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
        )

    def filter_web_nlg_lexicalizations(
        self, datasets, target_lang="cs", lex_source="deepl"
    ):
        def _preprocess_function(examples):
            new_lexes = []
            for lexes in examples["lex"]:
                filtered_lexes = [
                    text
                    for text, lang, source in zip(
                        lexes["text"], lexes["lang"], lexes["source"]
                    )
                    if lang == target_lang and lex_source in source
                ]
                new_lexes.append(filtered_lexes)

            return {"lex": new_lexes}

        return datasets.map(
            _preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
        )

    def flatten_lexicalizations_web_nlg(self, dataset, only_first_lexicalization=False):
        def _preprocess_function(examples):
            inputs = []
            targets = []

            for input, lexes in zip(
                examples["input"],
                examples["lex"],
            ):
                for lex in lexes:
                    inputs.append(input)
                    targets.append(lex)
                    if only_first_lexicalization:
                        break

            return {"input": inputs, "target": targets}

        return dataset.map(
            _preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=dataset.column_names,
        )

    def generate_inputs_web_nlg(self, datasets):
        def _preprocess_function(examples):
            inputs = []

            for modified_triple_sets in examples["modified_triple_sets"]:

                assert len(modified_triple_sets["mtriple_set"]) == 1
                modifiedtriple = " - ".join(modified_triple_sets["mtriple_set"][0])
                inputs.append(modifiedtriple)

            return {"input": inputs}

        return datasets.map(
            _preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
        )

    # def rename_columns_web_nlg(self, datasets):
    #     return datasets.rename_column("lex", "target")

    # def rename_columns_web_nlg(self, datasets, target_lang="cs", lex_source="deepl"):
    #     def _preprocess_function(examples):
    #         inputs = []
    #         targets = []

    #         for modified_triple_sets, lexes in zip(
    #             examples["modified_triple_sets"],
    #             examples["lex"],
    #         ):

    #             assert len(modified_triple_sets["mtriple_set"]) == 1
    #             modifiedtriple = " - ".join(modified_triple_sets["mtriple_set"][0])

    #             filtered_lexes = [
    #                 text
    #                 for text, lang, source in zip(
    #                     lexes["text"], lexes["lang"], lexes["source"]
    #                 )
    #                 if lang == target_lang and lex_source in source
    #             ]
    #             for lex in filtered_lexes:
    #                 inputs.append(modifiedtriple)
    #                 targets.append(lex)

    #         return {"input": inputs, "target": targets}

    #     return datasets.map(
    #         _preprocess_function,
    #         batched=True,
    #         num_proc=self.preprocessing_num_workers,
    #         remove_columns=datasets["train"].column_names,
    #     )

    def select_dataset_subset(
        self, datasets, max_train_samples, max_eval_samples, max_predict_samples
    ):
        limits = {
            "train": max_train_samples,
            "validation": max_eval_samples,
            "test": max_predict_samples,
        }
        for split_name, args_limit in limits.items():
            if split_name in datasets and args_limit is not None:
                limit = min(len(datasets[split_name]), args_limit)
                datasets[split_name] = datasets[split_name].select(range(limit))

        return datasets

    def tokenize(
        self,
        datasets,
        tokenizer,
        padding,
        max_source_length,
        max_target_length,
        ignore_pad_token_for_loss,
    ):
        # assert tokenizer is not None

        def _preprocess_function(examples):
            inputs = examples["input"]
            targets = examples["target"]
            model_inputs = tokenizer(
                inputs,
                max_length=max_source_length,
                padding=padding,
                truncation=True,
            )

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_target_length,
                    padding=padding,
                    truncation=True,
                )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label]
                    for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        any_split = next(iter(datasets.keys()))
        column_names = datasets[any_split].column_names

        return datasets.map(
            _preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
        )
