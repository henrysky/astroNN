Contributor and Issue Reporting guide
=====================================

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other
method with the maintainer of this repository before making a change.

Submitting bug reports and feature requests
---------------------------------------------

Bug reports and feature requests should be submitted by creating an issue on https://github.com/henrysky/astroNN

Pull Request
----------------

#. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
#. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports,
   useful file locations and container parameters.
#. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
   The versioning scheme we use is http://dex.rstsemver.org/
#. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission
   to do that, you may request the second reviewer to merge it for you.

New Model Proposal guide
-----------------------------
astroNN acts as a platform to share astronomy-oriented neural networks, so you are welcome to

* Add to ``astroNN\models\__init__.py``
* Add a documentation page for the new model and add link it appropriately in ``docs\souce\index.rst``
* Add the new model to the tree diagram and API udner appropriate class in ``docs\souce\neuralnets\basic_usage.rst``