import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkMetricCardsComponent } from './benchmark-metric-cards.component';

describe('BenchmarkMetricCardsComponent', () => {
  let component: BenchmarkMetricCardsComponent;
  let fixture: ComponentFixture<BenchmarkMetricCardsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkMetricCardsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkMetricCardsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
