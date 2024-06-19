## Requirements

To clone:

```setup
git clone -b deep-learning https://github.com/bintangfjulio/thesis-classification-app.git
```

## Requirements

To install requirements:

```setup
!pip install -r requirements.txt
```

## Training

To train the model, run this command:

```train
!python trainer.py
```

or more advanced train:

```
!python trainer.py --hyperparameter_name=value
```

example:

```
!python trainer.py --batch_size=16 --bert_model=indolem/indobertweet-uncased
```

You can see the other hyperparameter on util/hyperparameter.py
