import subprocess as sub

print('Please wait, this may take a while')

#sub.check_call('nosetests --with-coverage --cover-package=wafo', stderr=sub.STDOUT)
sub.check_call('nosetests', stderr=sub.STDOUT)
print('Finished')