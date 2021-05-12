from django.shortcuts import render,redirect
from django.contrib import messages
from .forms import UserRegistrationForm,UserUpdateForm,ProfileUpdateForm
from django.contrib.auth.decorators import login_required

# Create your views here.

def register(request):
   
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            #username = form.cleaned_data.get('username')
            form.save()
            messages.success(request,f'Your Account has been Created, you can now login')
            return redirect('login')
        else:
            errors = form.errors
            messages.error(request,errors)
    else:
        form = UserRegistrationForm()

    context={
        'form':form,
        }
    return render(request, 'users/register.html',context)



@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST,instance=request.user)
        p_form = ProfileUpdateForm(request.POST,request.FILES,instance=request.user.profile)

        if u_form.is_valid() and u_form.is_valid():
            #username = form.cleaned_data.get('username')
            u_form.save()
            p_form.save()
            
            messages.success(request,f'Your Account has been Updated')
            return redirect('profile')
        else:
            errors = u_form.errors + p_form.errors
            messages.error(request,errors)

    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context={
        'u_form':u_form,
        'p_form':p_form
    }
    return render(request,'users/profile.html',context)
