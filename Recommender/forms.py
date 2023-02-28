from django import forms
from .models import Rating

RATINGS = [
        (5, '5'),
        (4, '4'),
        (3, '3'),
        (2, '2'),
        (1, '1'),
    ]

class RatingForm(forms.ModelForm):

    class Meta:   
        model = Rating
        fields = ['rating']
        widgets = {'rating': forms.RadioSelect}
 

    # def clean(self):
    #     print('clean')
    #     if self.cleaned_data.get('rating')==0:
    #         raise forms.ValidationError('No name!')
    #     return self.cleaned_data  

    # def save(self, *args, **kwargs):
    #     self.full_clean()
    #     return super().save(*args, **kwargs)       

    # rate = forms.IntegerField(widget=forms.RadioSelect(choices=RATINGS))