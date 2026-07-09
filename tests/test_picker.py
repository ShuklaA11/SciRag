"""SB-B2: domain-picker presenter. Guards that core hands the view every
registered domain with the display metadata pulled faithfully from the profile,
so the UI never touches DomainProfile internals."""

from __future__ import annotations

from src.domain import available, get_profile
from src.hub import DomainOption, domain_options


def test_options_cover_every_registered_domain():
    assert [o.name for o in domain_options()] == available()


def test_options_are_sorted_by_name():
    names = [o.name for o in domain_options()]
    assert names == sorted(names)


def test_option_fields_mirror_the_profile():
    opt = next(o for o in domain_options() if o.name == "nlp_ml")
    profile = get_profile("nlp_ml")
    assert opt == DomainOption(
        name=profile.name,
        embedder=profile.embedder_name,
        verification_strategy=profile.verification_strategy,
        eval_benchmark=profile.eval_benchmark,
        data_sources=profile.data_sources,
    )


def test_biomedical_option_present_and_distinct():
    opts = {o.name: o for o in domain_options()}
    assert {"nlp_ml", "biomedical"} <= opts.keys()
    assert opts["nlp_ml"] != opts["biomedical"]
