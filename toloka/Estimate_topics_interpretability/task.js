exports.Task = extend(TolokaHandlebarsTask, function(options) {
    TolokaHandlebarsTask.call(this, options);
}, {
    is_good_topic: function(solution) {
        const positives = ['good', 'rather_good']
        return positives.includes(solution.output_values['quality'])
    },

    intersection: function(setA, setB) {
        return new Set(
            [...setA].filter(element => setB.has(element))
        );
    },

    union: function(setA, setB) {
        return new Set(
            [...setA, ...setB]
        );
    },

    bad_words_set: function(obj) {
        let badWordsSet = new Set();
        for (var prop in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, prop)) {
                let word = prop;
                let is_bad = obj[word];
                if (is_bad) {
                    badWordsSet.add(word);   
                }
            }
        }

        return badWordsSet;
    },

    setSolution: function(solution) {
        TolokaHandlebarsTask.prototype.setSolution.apply(this, arguments);
        var workspaceOptions = this.getWorkspaceOptions();

        var tname = solution.output_values['topic_name'] || "";
        this.setSolutionOutputValue("topic_name", tname);

        if (this.rendered) {
            if (!workspaceOptions.isReviewMode && !workspaceOptions.isReadOnly) {
                // Show a set of checkboxes if the answer "There are violations" (BAD) is selected. Otherwise, hide it
                if (solution.output_values['quality']) {
                    
                    var row = this.getDOMElement().querySelector('.second_scale');
                    row.style.display = this.is_good_topic(solution) ? 'block' : 'none';

                    if (!this.is_good_topic(solution)) {
                        let data = this.getTemplateData();
                        let words_out = {};
                        for (let i = 0; i < data.words.length; i++) {
                            words_out[data.words[i].name] = false;
                        }

                        this.setSolutionOutputValue("bad_words", words_out);

                        this.setSolutionOutputValue("topic_name", "");

                    }
                }
            }
        }
    },

    getTemplateData: function() {
        let data = TolokaHandlebarsTask.prototype.getTemplateData.call(this);

        const words = data.wordset.split(" ");
        let word_outs = [];
        for (let i = 0; i < words.length; i++) {
            word_outs.push({'name': words[i], 'title': words[i]});
        }

        data.words = word_outs;

        return data;
    },

    // Error message processing
    addError: function(message, field, errors) {
        errors || (errors = {
            task_id: this.getOptions().task.id,
            errors: {}
        });
        errors.errors[field] = {
            message: message
        };

        return errors;
    },

    // Checking the answers: if the answer "There are violations" is selected, at least one checkbox must be checked
    validate: function(solution) {
        var errors = null;
        var topic_name = solution.output_values.topic_name;
        topic_name = typeof topic_name !== 'undefined' ? topic_name.trim() : "";
        let bad_topic_name = topic_name.length < 3 || topic_name.length > 50 

        if (this.is_good_topic(solution) && bad_topic_name) {
            errors = this.addError("Topic name is less than 3 symbols or more than 50", '__TASK__', errors);
        }

        var correctBadWords = this.getTask().input_values.correct_bad_words;
        var golden;
        if (!correctBadWords) {
            golden = false;
        } else {
            var badWords = solution.output_values.bad_words;

            let correctBadWordsSet = this.bad_words_set(correctBadWords);
            let badWordsSet = this.bad_words_set(badWords);

            var intersection = this.intersection(correctBadWordsSet, badWordsSet) ;
            var union = this.union(correctBadWordsSet, badWordsSet);
            var golden = intersection.size / union.size >= 0.8 ? true : false;
        }
        this.setSolutionOutputValue("golden_bad_words", golden);

        var goldenBinaryQuality = this.is_good_topic(solution);
        this.setSolutionOutputValue("golden_binary_quality", goldenBinaryQuality); 

        return errors || TolokaHandlebarsTask.prototype.validate.apply(this, arguments);
    },

    // Open the second question block in verification mode to see the checkboxes marked by the performer
    onRender: function() {
        var workspaceOptions = this.getWorkspaceOptions();

        if (workspaceOptions.isReviewMode || workspaceOptions.isReadOnly || this.is_good_topic(this.getSolution())){
            var row = this.getDOMElement().querySelector('.second_scale');
            row.style.display = 'block';
        }

        this.rendered = true;
    }
});

function extend(ParentClass, constructorFunction, prototypeHash) {
    constructorFunction = constructorFunction || function() {
    };
    prototypeHash = prototypeHash || {};
    if (ParentClass) {
        constructorFunction.prototype = Object.create(ParentClass.prototype);
    }
    for (var i in prototypeHash) {
        constructorFunction.prototype[i] = prototypeHash[i];
    }
    return constructorFunction;
}
